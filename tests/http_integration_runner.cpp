#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#if defined(__unix__) || defined(__APPLE__)
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

#ifndef RUNTIME_SERVER_BIN
#define RUNTIME_SERVER_BIN "runtime-server"
#endif

namespace {

int failures = 0;

void check(bool condition, const char *message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

std::string shell_request(unsigned short port, const std::string &path,
                          const std::string &extra_args = "") {
    const std::string cmd = "curl -sS -m 15 --connect-timeout 2 " + extra_args +
                            " http://127.0.0.1:" + std::to_string(port) + path;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd.c_str(), "r"), pclose);
    if (!pipe) {
        throw std::runtime_error("failed to run curl");
    }
    std::string output;
    char buffer[256];
    while (fgets(buffer, sizeof(buffer), pipe.get()) != nullptr) {
        output += buffer;
    }
    return output;
}

class SubprocessServer {
  public:
    explicit SubprocessServer(unsigned short port) : port_(port) {
#if defined(__unix__) || defined(__APPLE__)
        const pid_t pid = fork();
        if (pid < 0) {
            throw std::runtime_error("fork failed");
        }
        if (pid == 0) {
            const std::string port_arg = std::to_string(port_);
            freopen("/dev/null", "w", stdout);
            execl(RUNTIME_SERVER_BIN, RUNTIME_SERVER_BIN, "--host", "127.0.0.1", "--port",
                  port_arg.c_str(), nullptr);
            _exit(127);
        }
        pid_ = pid;
        usleep(200000);
#else
        (void)port_;
        throw std::runtime_error("POSIX required for HTTP integration runner");
#endif
    }

    ~SubprocessServer() {
#if defined(__unix__) || defined(__APPLE__)
        if (pid_ > 0) {
            kill(pid_, SIGTERM);
            waitpid(pid_, nullptr, 0);
        }
#endif
    }

    unsigned short port() const {
        return port_;
    }

  private:
    unsigned short port_{0};
    pid_t pid_{-1};
};

unsigned short pick_free_port() {
    const std::string cmd =
        "python3 -c \"import "
        "socket;s=socket.socket();s.bind(('127.0.0.1',0));print(s.getsockname()[1]);s.close()\"";
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(cmd.c_str(), "r"), pclose);
    if (!pipe) {
        throw std::runtime_error("failed to allocate port");
    }
    char buffer[32];
    if (fgets(buffer, sizeof(buffer), pipe.get()) == nullptr) {
        throw std::runtime_error("failed to read allocated port");
    }
    return static_cast<unsigned short>(std::stoi(buffer));
}

} // namespace

int main() {
    try {
        const unsigned short port = pick_free_port();
        SubprocessServer server(port);
        check(server.port() > 0, "server port assigned");

        const auto health = shell_request(server.port(), "/health", "-i");
        check(health.find("200") != std::string::npos, "health status");
        check(health.find("healthy") != std::string::npos, "health body");

        const std::string body = R"({"model_id":"model-a","prompt":"hello"})";
        const auto infer =
            shell_request(server.port(), "/v1/infer",
                          "-i -X POST -H 'Content-Type: application/json' -d '" + body + "'");
        check(infer.find("200") != std::string::npos, "infer status");
        check(infer.find("Completed") != std::string::npos, "infer completed");

        const auto missing = shell_request(server.port(), "/v1/infer",
                                           "-i -X POST -H 'Content-Type: application/json' -d "
                                           "'{\"model_id\":\"missing\",\"prompt\":\"hello\"}'");
        check(missing.find("404") != std::string::npos, "unknown model status");

        const auto method = shell_request(server.port(), "/health", "-i -X DELETE");
        check(method.find("405") != std::string::npos, "unsupported method");

        const auto stream =
            shell_request(server.port(), "/v1/infer/stream",
                          "-i -N -X POST -H 'Content-Type: application/json' -d '" + body + "'");
        check(stream.find("200") != std::string::npos, "stream status");
        check(stream.find("\"event\":\"state\"") != std::string::npos, "stream state event");
        check(stream.find("\"event\":\"terminal\"") != std::string::npos, "stream terminal event");
    } catch (const std::exception &ex) {
        std::cerr << "FAIL: exception: " << ex.what() << '\n';
        return 1;
    }

    if (failures != 0) {
        std::cerr << failures << " HTTP integration failure(s)\n";
        return 1;
    }
    std::cout << "HTTP integration runner passed\n";
    return 0;
}
