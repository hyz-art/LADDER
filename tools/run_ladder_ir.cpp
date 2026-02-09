#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>

#include "gemm.h"

namespace {

std::string trim(const std::string& s) {
    size_t b = s.find_first_not_of(" \t\n\r");
    size_t e = s.find_last_not_of(" \t\n\r");
    if (b == std::string::npos) return "";
    return s.substr(b, e - b + 1);
}

std::unordered_map<std::string,std::string> parse_attrs(const std::string& content) {
    std::unordered_map<std::string,std::string> attrs;
    auto pos = content.find("ladder.gemm");
    if (pos == std::string::npos) return attrs;
    auto lbrace = content.find('{', pos);
    auto rbrace = content.find('}', lbrace);
    if (lbrace == std::string::npos || rbrace == std::string::npos) return attrs;
    auto inside = content.substr(lbrace + 1, rbrace - lbrace - 1);
    std::stringstream ss(inside);
    std::string token;
    while (std::getline(ss, token, ',')) {
        auto eq = token.find('=');
        if (eq == std::string::npos) continue;
        std::string key = trim(token.substr(0, eq));
        std::string val = trim(token.substr(eq + 1));
        attrs[key] = val;
    }
    return attrs;
}

ladder::tType::Precision parse_prec(const std::string& p) {
    std::string v = p;
    for (auto &c : v) c = std::tolower(c);
    if (v == "fp16" || v == "f16") return ladder::tType::Precision::FP16;
    if (v == "fp8" || v == "f8") return ladder::tType::Precision::FP8;
    if (v == "int8" || v == "i8") return ladder::tType::Precision::INT8;
    if (v == "int4" || v == "i4") return ladder::tType::Precision::INT4;
    return ladder::tType::Precision::FP32;
}

ladder::GemmImpl parse_impl(const std::string& v) {
    std::string s = v;
    for (auto &c : s) c = std::tolower(c);
    if (s == "ttile") return ladder::GemmImpl::TTile;
    return ladder::GemmImpl::Loop;
}

size_t get_size(const std::unordered_map<std::string,std::string>& a, const std::string& k, size_t defv) {
    auto it = a.find(k);
    if (it == a.end()) return defv;
    return static_cast<size_t>(std::stoul(it->second));
}

float get_float(const std::unordered_map<std::string,std::string>& a, const std::string& k, float defv) {
    auto it = a.find(k);
    if (it == a.end()) return defv;
    return std::stof(it->second);
}

int get_int(const std::unordered_map<std::string,std::string>& a, const std::string& k, int defv) {
    auto it = a.find(k);
    if (it == a.end()) return defv;
    return std::stoi(it->second);
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: run_ladder_ir <ladder_ir.mlir>\n";
        return 1;
    }

    std::ifstream in(argv[1]);
    if (!in) {
        std::cerr << "Failed to open: " << argv[1] << "\n";
        return 1;
    }
    std::stringstream buffer;
    buffer << in.rdbuf();
    auto attrs = parse_attrs(buffer.str());

    size_t M = get_size(attrs, "M", 128);
    size_t N = get_size(attrs, "N", 128);
    size_t K = get_size(attrs, "K", 64);
    size_t tileM = get_size(attrs, "tileM", 64);
    size_t tileN = get_size(attrs, "tileN", 64);
    size_t tileK = get_size(attrs, "tileK", 16);
    auto prec = parse_prec(attrs.count("precision") ? attrs["precision"] : "fp32");
    auto acc = parse_prec(attrs.count("accumulate") ? attrs["accumulate"] : "fp32");
    auto impl = parse_impl(attrs.count("impl") ? attrs["impl"] : "loop");
    float in_scale = get_float(attrs, "input_scale", 1.0f);
    float out_scale = get_float(attrs, "output_scale", 1.0f);
    int fuse_relu = get_int(attrs, "fuse_relu", 0);

    std::vector<float> A(M*K), B(K*N), C(M*N);
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1.0f,1.0f);
    for (auto &v : A) v = dist(rng);
    for (auto &v : B) v = dist(rng);

    ladder::QuantParams q;
    q.compute = prec;
    q.accumulate = acc;
    q.input_scale = in_scale;
    q.output_scale = out_scale;

    std::cout << "Running GEMM M="<<M<<" N="<<N<<" K="<<K
              <<" tile("<<tileM<<","<<tileN<<","<<tileK<<") impl="<<(impl==ladder::GemmImpl::TTile?"ttile":"loop")
              <<" precision="<<(int)prec<<" accumulate="<<(int)acc<<"\n";

    if (impl == ladder::GemmImpl::TTile) {
        ladder::gemm_tiled_fused_tiles_quantized(M,N,K,A,B,C,tileM,tileN,tileK,q,nullptr,fuse_relu!=0);
    } else {
        ladder::gemm_tiled_fused(M,N,K,A,B,C,tileM,tileN,tileK,impl,nullptr,fuse_relu!=0,prec);
    }

    std::cout << "Done. C[0]="<<C[0]<<" C[last]="<<C.back()<<"\n";
    return 0;
}
