// src/ds41/tests/pack_format_test.cpp - the pack's expert format (pack_info.txt expert_format) and how the loader
// maps DeepSeek's FP4 components onto the expert slot. No GPU.
#include "fake_pack.hpp"
#include "strata/ds41/pack.hpp"

#include <cstdio>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

using namespace strata::ds41;

static int failures = 0;
#define CHECK(c)                                                              \
    do {                                                                      \
        if (!(c)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #c); \
            ++failures;                                                       \
        }                                                                     \
    } while (0)

static std::string tmpdir(const char* tag) {
    char t[] = "/tmp/ds41_pack_format_XXXXXX";
    std::string d = mkdtemp(t);
    return d + "/" + tag;
}

static bool throws(const std::string& dir) {
    try {
        Pack p(dir);
    } catch (const std::runtime_error&) {
        return true;
    }
    return false;
}

/// replace the pack_info line that starts with `key` (or add it before "finished")
static void set_info(const std::string& dir, const std::string& key, const std::string& value) {
    std::ifstream in(dir + "/pack_info.txt");
    std::stringstream out;
    std::string line;
    bool done = false;
    while (std::getline(in, line)) {
        if (line.rfind(key + " ", 0) == 0) continue;
        if (line == "finished 1" && !done) {
            out << key << " " << value << "\n";
            done = true;
        }
        out << line << "\n";
    }
    std::ofstream(dir + "/pack_info.txt") << out.str();
}

int main() {
    {   // an EXL3 pack's dense tensors pass the shape checks; a wrong trellis shape fails them
        const std::string d = tmpdir("exl3dense");
        ds41test::write_fake_pack(d, false, 256, false, true);
        Pack p(d);
        CHECK(p.has_dense("layers.0.attn.wq_b.trellis") && !p.has_dense("layers.0.attn.wq_b.weight"));
        CHECK(p.has_dense("head.trellis") && !p.has_dense("head.weight"));
        std::ifstream in(d + "/index.txt");
        std::stringstream out;
        std::string line;
        while (std::getline(in, line)) {
            if (line.rfind("layers.0.attn.wq_b.trellis ", 0) == 0) line = "layers.0.attn.wq_b.trellis i16 3 80 1024 80 0 0";
            out << line << "\n";
        }
        std::ofstream(d + "/index.txt") << out.str();
        CHECK(throws(d));
    }
    {   // an old pack has no expert_format line: EXL3, components as before
        const std::string d = tmpdir("exl3");
        ds41test::write_fake_pack(d);
        Pack p(d);
        CHECK(p.expert_format() == ExpertFormat::Exl3);
        const ExpertSlot& x = p.expert(3, 7);
        for (int c = 0; c < 12; ++c) CHECK(x.comp_off[c] == (uint64_t) c * 256 && x.comp_bytes[c] == 256);
    }
    {   // FP4: weight in the trellis place, the scale in both scale places, nothing in the fourth
        const std::string d = tmpdir("mxfp4");
        ds41test::write_fake_pack(d, false, 256, true);
        Pack p(d);
        CHECK(p.expert_format() == ExpertFormat::Mxfp4);
        const ExpertSlot& x = p.expert(39, 383);
        for (int w = 0; w < 3; ++w) {
            const int c0 = 4 * w;
            CHECK(x.comp_off[c0] == (uint64_t) (2 * w) * 256 && x.comp_bytes[c0] == 256);
            CHECK(x.comp_off[c0 + 1] == (uint64_t) (2 * w + 1) * 256 && x.comp_bytes[c0 + 1] == 256);
            CHECK(x.comp_off[c0 + 2] == x.comp_off[c0 + 1] && x.comp_bytes[c0 + 2] == x.comp_bytes[c0 + 1]);
            CHECK(x.comp_bytes[c0 + 3] == 0);
            CHECK(x.bits[w] == 4.25f);
        }
    }
    {   // the format and the components must agree
        const std::string d = tmpdir("mismatch");
        ds41test::write_fake_pack(d, false, 256, true);
        set_info(d, "expert_format", "exl3");
        CHECK(throws(d));
        const std::string d2 = tmpdir("mismatch2");
        ds41test::write_fake_pack(d2);
        set_info(d2, "expert_format", "mxfp4");
        CHECK(throws(d2));
    }
    {   // an unknown format is refused
        const std::string d = tmpdir("unknown");
        ds41test::write_fake_pack(d);
        set_info(d, "expert_format", "nvfp4");
        CHECK(throws(d));
    }
    if (failures) return 1;
    std::printf("pack_format_test: OK\n");
    return 0;
}
