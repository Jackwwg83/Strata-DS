// Synthetic CLI coverage. This executable does not link the CUDA engine.
#define main ds41_cli_main
#include "../ds41_generate.cpp"
#undef main
#include <unistd.h>

namespace {
int windows = 0, multi = 0, rejected = 0;
std::string last_text;   // the whole stdout of the last cli() call
constexpr int sequence[] = {101, 102, 103, 104, 101, 102, 103, 105};
int predict(int pos) { return sequence[pos % 8]; }
void check(bool value) { if (!value) throw std::runtime_error("mock generation assertion"); }
}
namespace strata::ds41 {
struct Engine::Impl {
    std::vector<int> history, pending;
    int max_seq;
    std::vector<float> logits;
};
Engine::Engine(const std::string&, const EngineOptions& options) : impl_(new Impl) { impl_->max_seq = options.max_seq; }
Engine::~Engine() = default;
int Engine::step(int token, int pos, StepDump*) {
    check(impl_->pending.empty() && pos == int(impl_->history.size()) && pos < impl_->max_seq);
    impl_->history.push_back(token);
    return predict(pos);
}
int Engine::prefill(const std::vector<int>& tokens, int pos, std::vector<float>*) {
    int next = -1;
    for (int token : tokens) next = step(token, pos++);
    return next;
}
VerifyResult Engine::verify(const std::vector<int>& input, int pos, bool) {
    check(impl_->pending.empty() && pos == int(impl_->history.size()));
    check(!input.empty() && input.size() <= kVerifyMaxTokens && pos+int(input.size()) <= impl_->max_seq);
    ++windows; multi += input.size() > 1;
    impl_->pending = input;
    VerifyResult result;
    for (size_t i = 0; i < input.size(); ++i) result.next.push_back(predict(pos+int(i)));
    return result;
}
void Engine::commit(int keep) {
    check(keep > 0 && keep <= int(impl_->pending.size()));
    rejected += keep < int(impl_->pending.size());
    impl_->history.insert(impl_->history.end(), impl_->pending.begin(), impl_->pending.begin()+keep);
    impl_->pending.clear();
}
int Engine::vram_expert_slots() const { return 0; }
const std::vector<float>& Engine::last_logits() const { return impl_->logits; }
} // namespace strata::ds41

static std::vector<int> cli(std::vector<std::string> args) {
    std::vector<char*> argv;
    for (auto& a : args) argv.push_back(a.data());
    std::FILE* output = std::tmpfile();
    check(output != nullptr);
    std::fflush(stdout);
    const int saved = dup(STDOUT_FILENO);
    check(saved >= 0 && dup2(fileno(output), STDOUT_FILENO) >= 0);
    const int status = ds41_cli_main(int(argv.size()), argv.data());
    std::fflush(stdout);
    check(dup2(saved, STDOUT_FILENO) >= 0); close(saved);
    check(status == 0);
    std::rewind(output);
    std::string text;
    char buffer[4096];
    while (size_t n = std::fread(buffer, 1, sizeof buffer, output)) text.append(buffer, n);
    std::fclose(output);
    last_text = text;
    const size_t at = text.find("generated:");
    check(at != std::string::npos);
    std::istringstream line(text.substr(at+10, text.find('\n', at)-at-10));
    std::vector<int> tokens;
    int token; while (line >> token) tokens.push_back(token);
    return tokens;
}
int main() {
    try {
        for (bool prefill : {false, true}) for (int count : {0, 1, 2, 3, 4, 5, 31, 64}) {
            std::vector<std::string> args{"mock", "--pack", "synthetic", "--ids", "10", "--gen", std::to_string(count)};
            if (prefill) args.push_back("--prefill");
            const auto plain = cli(args);
            args.insert(args.end(), {"--spec", "suffix", "--spec-max", "4"});
            const auto spec = cli(args);
            check(plain == spec && int(spec.size()) == count);
            for (int i = 0; i < count; ++i) check(spec[i] == predict(i));
        }
        // the speed scripts (m3_context.sh) read these lines: keep them in the generation path
        cli({"mock", "--pack", "synthetic", "--ids", "10,11,12", "--gen", "8", "--prefill"});
        for (const char* key : {"prefill_tokens 3 ms", "chunk_tokens", "streamed", "stream_wait_ms",
                                "decode_ms_per_token", "hit_rate"})
            check(last_text.find(key) != std::string::npos);
        cli({"mock", "--pack", "synthetic", "--ids", "10", "--gen", "8", "--spec", "suffix"});
        check(last_text.find("decode_ms_per_token") != std::string::npos);
        for (int eos : {101, 104, 105}) {
            std::vector<std::string> args{"mock", "--pack", "synthetic", "--ids", "10", "--gen", "64", "--eos-id", std::to_string(eos)};
            const auto plain = cli(args);
            args.insert(args.end(), {"--spec", "suffix"});
            const auto spec = cli(args);
            check(plain == spec && spec.back() == eos);
        }
        const auto capped = cli({"mock", "--pack", "synthetic", "--ids", "10", "--gen", "64",
                                 "--spec", "suffix", "--spec-max", "8", "--max-seq", "64"});
        check(capped.size() == 64);
        for (int i = 0; i < 64; ++i) check(capped[i] == predict(i));
        check(windows > 0 && multi > 0 && rejected > 0);
        std::printf("RESULT pass generate_suffix_mock windows=%d multi=%d rejected=%d gen_bounds=1 eos=1 prefill=1\n",
                    windows, multi, rejected);
        return 0;
    } catch (const std::exception& e) { std::fprintf(stderr, "RESULT fail %s\n", e.what()); return 1; }
}
