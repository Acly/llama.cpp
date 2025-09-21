#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <stdexcept>
#include <array>
#include <vector>
#include <map>
#include <algorithm>
#include <filesystem>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cassert>

char const * usage = R"(Usage: vulkan-shaders-gen [options]

Compiles Vulkan compute shaders to SPIR-V and embeds it into C++ source files

Options:
  --glslc <path>          Path to glslc executable (default: glslc)
  --input-dir <path>      Input directory containing .comp shader files
  --output-dir <path>     Output directory for compiled .spv files
  --target-hpp <path>     Output C++ header file path
  --target-cpp <path>     Output C++ source file path
  --target-cmake <path>   Output CMakeLists.txt file path
  --no-embed              Do not embed SPIR-V binaries into C++ source

This executable runs at build time. Typically it is invoked by CMake like this:
  1. Run with --target-cmake to generate CMakeLists.txt that contains build
     commands for the shaders.
  2. Configure and build the generated CMake sub-project to compile the shaders
     into SPIR-V files.
  3. Run without --target-cmake to generate C++ source files that embed the
     SPIR-V binaries. This invocation is part of the generated sub-project.

If --no-embed is used, step 1 will generate stub C++ source files, and
step 3 is skipped. This allows fast iteration on shader code without
recompiling C++ code, but can't be deployed.
)";


using std::filesystem::path;

std::vector<std::pair<std::string, path>> shader_fnames;
std::locale c_locale("C");

std::string GLSLC = "glslc";
path input_dir = "vulkan-shaders";
path output_dir = "/tmp";

const std::vector<std::string> type_names = {
    "f32",
    "f16",
    "q4_0",
    "q4_1",
    "q5_0",
    "q5_1",
    "q8_0",
    "q2_k",
    "q3_k",
    "q4_k",
    "q5_k",
    "q6_k",
    "iq1_s",
    "iq1_m",
    "iq2_xxs",
    "iq2_xs",
    "iq2_s",
    "iq3_xxs",
    "iq3_s",
    "iq4_xs",
    "iq4_nl",
    "mxfp4",
    "bf16",
};

enum MatMulIdType {
    NONE,
    DEFAULT,
    SUBGROUP,
};

namespace {

std::string to_uppercase(const std::string& input) {
    std::string result = input;
    for (char& c : result) {
        c = std::toupper(c);
    }
    return result;
}

bool string_starts_with(const std::string& str, const std::string& prefix) {
    if (prefix.size() > str.size()) {
        return false;
    }
    return std::equal(prefix.begin(), prefix.end(), str.begin());
}

bool string_ends_with(const std::string& str, const std::string& suffix) {
    if (suffix.size() > str.size()) {
        return false;
    }
    return std::equal(suffix.rbegin(), suffix.rend(), str.rbegin());
}

bool is_quantized_type(const std::string& type_name) {
    return type_name != "f32" && type_name != "f16" && type_name != "bf16";
}

bool is_legacy_quant(const std::string& type_name) {
    return type_name == "q4_0" || type_name == "q4_1" || type_name == "q5_0" || type_name == "q5_1" || type_name == "q8_0";
}

bool is_k_quant(const std::string& type_name) {
    return string_ends_with(type_name, "_k");
}

bool is_iq_quant(const std::string& type_name) {
    return string_starts_with(type_name, "iq");
}

std::stringstream make_generic_stringstream() {
    std::stringstream ss;
    ss.imbue(c_locale);
    return ss;
}

std::vector<unsigned char> read_binary_file(const path& path, bool may_not_exist = false) {
    std::fstream f(path, std::ios::in | std::ios::binary);
    if (!f) {
        if (!may_not_exist) {
            std::cerr << "Error opening file: " << path << " (" << strerror(errno) << ")\n";
        }
        return {};
    }

    f.seekg(0, std::ios::end);
    size_t size = f.tellg();
    f.seekg(0, std::ios::beg);

    std::vector<unsigned char> data(size);
    f.read(reinterpret_cast<char*>(data.data()), size);
    if (!f) {
        std::cerr << "Error reading file: " << path << " (" << strerror(errno) << ")\n";
        return {};
    }
    return data;
}

void write_binary_file(const path& path, const unsigned char * data, size_t size) {
    std::fstream f(path, std::ios::out | std::ios::binary);
    if (!f) {
        std::cerr << "Error opening file for writing: " << path << " (" << strerror(errno) << ")\n";
        return;
    }

    f.write(reinterpret_cast<const char*>(data), size);
    if (!f) {
        std::cerr << "Error writing file: " << path << " (" << strerror(errno) << ")\n";
        return;
    }
}

void write_binary_file(const path& path, const std::string& content) {
    write_binary_file(path, (const unsigned char *)content.data(), content.size());
}

void write_file_if_changed(const path& path, const std::string& content) {
    std::vector<unsigned char> existing = read_binary_file(path, true);
    if (existing.size() != content.size() || memcmp(existing.data(), content.data(), content.size()) != 0) {
        write_binary_file(path, content);
    }
}

struct cmake_escape { const std::string& str; };
std::ostream& operator<<(std::ostream& os, const cmake_escape& to_escape) {
    for (char c : to_escape.str) {
        if (c == '"' || c == '\\') {
            os << '\\';
        }
        os << c;
    }
    return os;
}

struct cmake_lists {
    std::stringstream out = make_generic_stringstream();
    std::vector<path> out_filepaths;

    void add_header(int argc, char ** argv) {
        out << "# Generated with ";
        for (int i = 0; i < argc; i++) {
            out << argv[i] << " ";
        }
        out << "\n\n";
        out << "cmake_minimum_required(VERSION 3.14)\n";
        out << "project(ggml-vulkan-shaders)\n\n";
        out << "set(GLSLC \"" << GLSLC << "\")\n\n";
        out << "function(compile_shader name in_file out_file flags)\n";
        out << "  add_custom_command(\n";
        out << "    OUTPUT ${out_file}\n";
        out << "    COMMAND ${GLSLC} ${flags} ${ARGN} -MD -MF ${out_file}.d ${in_file} -o ${out_file}\n";
        out << "    DEPENDS ${in_file}\n";
        out << "    DEPFILE ${out_file}.d\n";
        out << "    COMMENT \"Building Vulkan shader ${name}.spv\"\n";
        out << "  )\n";
        out << "endfunction()\n\n";
    }

    void add_build_command(std::string_view name, const path & in_filepath, const path & out_filepath, const std::vector<std::string> & flags) {
        out << "compile_shader(" << name << " " << in_filepath << " " << out_filepath << " ";
        for (const std::string & flag : flags) {
            out << "\"" << cmake_escape{ flag } << "\" ";
        }
        out << ")\n";
        out_filepaths.emplace_back(out_filepath);
    }

    void add_target_embed(const path & shaders_gen_executable, const path & target_hpp, const path & target_cpp) {
        out << "\nadd_custom_command(\n";
        out << "  OUTPUT " << target_hpp << " " << target_cpp << "\n";
        out << "  COMMAND " << shaders_gen_executable 
            << " --glslc " << GLSLC
            << " --input-dir " << input_dir
            << " --output-dir " << output_dir
            << " --target-hpp " << target_hpp
            << " --target-cpp " << target_cpp << "\n";
        out << "  DEPENDS\n";
        for (const path & spv_path : out_filepaths) {
            out << "    " << spv_path << "\n";
        }
        out << "  COMMENT \"Embedding Vulkan shaders into C++ source\"\n";
        out << ")\n";

        out << "\nadd_custom_target(vulkan-shaders ALL DEPENDS\n";
        out << "  " << target_hpp << "\n";
        out << "  " << target_cpp << "\n";
        out << ")\n";
    }

    void add_target_build_only() {
        out << "\nadd_custom_target(vulkan-shaders ALL DEPENDS\n";
        for (const auto & spv_path : out_filepaths) {
            out << "  " << spv_path << "\n";
        }
        out << ")\n";
    }

    void write(const path & target_filepath) { write_file_if_changed(target_filepath, out.str()); }
};

cmake_lists cmake;

std::map<std::string, std::string> merge_maps(const std::map<std::string, std::string>& a, const std::map<std::string, std::string>& b) {
    std::map<std::string, std::string> result = a;
    result.insert(b.begin(), b.end());
    return result;
}

void string_to_spv(const std::string& _name, const std::string& in_fname, const std::map<std::string, std::string>& defines, bool fp16 = true, bool coopmat = false, bool coopmat2 = false, bool f16acc = false) {
    std::string name = _name + (f16acc ? "_f16acc" : "") + (coopmat ? "_cm1" : "") + (coopmat2 ? "_cm2" : (fp16 ? "" : "_fp32"));
    path out_path = output_dir / (name + ".spv");
    path in_path = input_dir / in_fname;

    std::string target_env = (name.find("_cm2") != std::string::npos) ? "--target-env=vulkan1.3" : "--target-env=vulkan1.2";

    // disable spirv-opt for coopmat shaders for https://github.com/ggerganov/llama.cpp/issues/10734
    // disable spirv-opt for bf16 shaders for https://github.com/ggml-org/llama.cpp/issues/15344
    std::string opt_level = (coopmat || name.find("bf16") != std::string::npos) ? "" : "-O";

    std::vector<std::string> flags = {"-fshader-stage=compute", target_env, opt_level};

    #ifdef GGML_VULKAN_SHADER_DEBUG_INFO
        flags.push_back("-g");
    #endif

    for (const auto& define : defines) {
        flags.push_back("-D" + define.first + "=" + define.second);
    }

    cmake.add_build_command(name, in_path, out_path, flags);
    shader_fnames.emplace_back(name, out_path);
}

void matmul_shaders(bool fp16, MatMulIdType matmul_id_type, bool coopmat, bool coopmat2, bool f16acc) {
    std::string load_vec = coopmat2 ? "1" : fp16 ? "8" : "4";
    std::string aligned_b_type_f32 = coopmat2 ? "float" : fp16 ? "mat2x4" : "vec4";
    std::string aligned_b_type_f16 = coopmat2 ? "float16_t" : fp16 ? "f16mat2x4" : "f16vec4";

    std::map<std::string, std::string> base_dict = {
        {"FLOAT_TYPE_VEC2", (coopmat2 || fp16) ? "f16vec2" : "vec2"},
    };
    std::string shader_name = "matmul";

    if (matmul_id_type == MatMulIdType::DEFAULT) {
        base_dict["MUL_MAT_ID"] = "1";
        shader_name = "matmul_id";
    } else if (matmul_id_type == MatMulIdType::SUBGROUP) {
        base_dict["MUL_MAT_ID"] = "1";
        base_dict["MUL_MAT_ID_USE_SUBGROUPS"] = "1";
        shader_name = "matmul_id_subgroup";
    }

    if (fp16) {
        base_dict["FLOAT16"] = "1";
    }

    base_dict["ACC_TYPE"] = f16acc ? "float16_t" : "float";
    if (f16acc) {
        base_dict["ACC_TYPE_MAX"] = "\"float16_t(65504.0)\"";
    }

    if (coopmat) {
        base_dict["COOPMAT"] = "1";
    }

    const std::string source_name = coopmat2 ? "mul_mm_cm2.comp" : "mul_mm.comp";

    auto const &FLOAT_TYPE = [&](const std::string &t) -> std::string {
        if (t == "bf16") {
            // scalar path promotes to float
            if (!coopmat && !coopmat2) {
                return "float";
            }
            return "bfloat16_t";
        }
        if (coopmat2 || fp16) {
            return "float16_t";
        }
        return "float";
    };

    // Shaders with f16 B_TYPE
    string_to_spv(shader_name + "_f32_f16",         source_name, merge_maps(base_dict, {{"FLOAT_TYPE", FLOAT_TYPE("f16")}, {"DATA_A_F32", "1"},                                                     {"B_TYPE", "float16_t"},                                          {"D_TYPE", "float"}, }), fp16, coopmat, coopmat2, f16acc);
    string_to_spv(shader_name + "_f32_f16_aligned", source_name, merge_maps(base_dict, {{"FLOAT_TYPE", FLOAT_TYPE("f16")}, {"DATA_A_F32", "1"}, {"LOAD_VEC_A", load_vec}, {"LOAD_VEC_B", load_vec}, {"B_TYPE", aligned_b_type_f16}, {"B_TYPE32", aligned_b_type_f32}, {"D_TYPE", "float"}, {"ALIGNED", "1"}}), fp16, coopmat, coopmat2, f16acc);

    string_to_spv(shader_name + "_f16_aligned",     source_name, merge_maps(base_dict, {{"FLOAT_TYPE", FLOAT_TYPE("f16")}, {"DATA_A_F16", "1"}, {"LOAD_VEC_A", load_vec}, {"LOAD_VEC_B", load_vec}, {"B_TYPE", aligned_b_type_f16}, {"B_TYPE32", aligned_b_type_f32}, {"D_TYPE", "float"}, {"ALIGNED", "1"}}), fp16, coopmat, coopmat2, f16acc);
    string_to_spv(shader_name + "_f16",             source_name, merge_maps(base_dict, {{"FLOAT_TYPE", FLOAT_TYPE("f16")}, {"DATA_A_F16", "1"},                                                     {"B_TYPE", "float16_t"},                                          {"D_TYPE", "float"}}), fp16, coopmat, coopmat2, f16acc);

    // bf16
    {
        std::string load_vec_a_unaligned = "1";
        // For aligned matmul loads
        std::string load_vec_a = coopmat2 ? "1" : "4";

        // scalar path promotes to float
        std::string to_float_type = (coopmat || coopmat2) ? "uintBitsToBFloat16EXT" : "bf16_to_fp32";

        // If bfloat16 is not supported, then only compile the scalar (promote to fp32) shader
#if !defined(GGML_VULKAN_BFLOAT16_GLSLC_SUPPORT)
        if (!(coopmat || coopmat2))
#endif
        {
            string_to_spv(shader_name + "_bf16_aligned", source_name, merge_maps(base_dict, {{"FLOAT_TYPE", FLOAT_TYPE("bf16")}, {"TO_FLOAT_TYPE", to_float_type}, {"DATA_A_BF16", "1"}, {"LOAD_VEC_A", load_vec_a},           {"LOAD_VEC_B", "4"}, {"B_TYPE", coopmat2 ? "bfloat16_t" : "u16vec4"},   {"B_TYPE32", "vec4"}, {"D_TYPE", "float"}, {"B_IS_FLOAT", "1"}, {"DATA_B_BF16", "1"}, {"ALIGNED", "1"}}), fp16, coopmat, coopmat2, f16acc);
            string_to_spv(shader_name + "_bf16",         source_name, merge_maps(base_dict, {{"FLOAT_TYPE", FLOAT_TYPE("bf16")}, {"TO_FLOAT_TYPE", to_float_type}, {"DATA_A_BF16", "1"}, {"LOAD_VEC_A", load_vec_a_unaligned},                      {"B_TYPE", coopmat2 ? "bfloat16_t" : "uint16_t"},                        {"D_TYPE", "float"}, {"B_IS_FLOAT", "1"}, {"DATA_B_BF16", "1"}}),                   fp16, coopmat, coopmat2, f16acc);
        }
    }

    for (const auto& tname : type_names) {
        std::string load_vec_quant = "2";
        if ((tname == "q4_0") || (tname == "q4_1") || (tname == "iq1_s") || (tname == "iq1_m") || (tname == "iq2_xxs") || (tname == "iq2_xs") || (tname == "iq2_s"))
            load_vec_quant = "8";
        else if ((tname == "q5_0") || (tname == "q5_1") || (tname == "q8_0") || (tname == "iq3_xxs") || (tname == "iq3_s") || (tname == "iq4_nl") || (tname == "mxfp4"))
            load_vec_quant = "4";

        if (tname == "bf16") {
            continue;
        }

        std::string data_a_key = "DATA_A_" + to_uppercase(tname);
        // For unaligned, load one at a time for f32/f16, or two at a time for quants
        std::string load_vec_a_unaligned = (coopmat2 || tname == "f32" || tname == "f16" || tname == "bf16") ? "1" : load_vec_quant;
        // For aligned matmul loads
        std::string load_vec_a = (coopmat2 || tname == "f32" || tname == "f16" || tname == "bf16") ? load_vec : load_vec_quant;

        // don't generate f32 variants for coopmat2
        if (!coopmat2) {
            string_to_spv(shader_name + "_" + tname + "_f32",         source_name, merge_maps(base_dict, {{"FLOAT_TYPE", FLOAT_TYPE(tname)}, {data_a_key, "1"}, {"LOAD_VEC_A", load_vec_a_unaligned},                           {"B_TYPE", "float"},                                              {"D_TYPE", "float"}}), fp16, coopmat, coopmat2, f16acc);
            string_to_spv(shader_name + "_" + tname + "_f32_aligned", source_name, merge_maps(base_dict, {{"FLOAT_TYPE", FLOAT_TYPE(tname)}, {data_a_key, "1"}, {"LOAD_VEC_A", load_vec_a},           {"LOAD_VEC_B", load_vec}, {"B_TYPE", aligned_b_type_f32}, {"B_TYPE32", aligned_b_type_f32}, {"D_TYPE", "float"}, {"ALIGNED", "1"}}), fp16, coopmat, coopmat2, f16acc);
        }

        if (tname != "f16" && tname != "f32") {
            string_to_spv(shader_name + "_" + tname + "_f16",         source_name,  merge_maps(base_dict, {{"FLOAT_TYPE", FLOAT_TYPE(tname)}, {data_a_key, "1"}, {"LOAD_VEC_A", load_vec_a_unaligned},                           {"B_TYPE", "float16_t"},                                          {"D_TYPE", "float"}}), fp16, coopmat, coopmat2, f16acc);
            string_to_spv(shader_name + "_" + tname + "_f16_aligned", source_name,  merge_maps(base_dict, {{"FLOAT_TYPE", FLOAT_TYPE(tname)}, {data_a_key, "1"}, {"LOAD_VEC_A", load_vec_a},           {"LOAD_VEC_B", load_vec}, {"B_TYPE", aligned_b_type_f16}, {"B_TYPE32", aligned_b_type_f32}, {"D_TYPE", "float"}, {"ALIGNED", "1"}}), fp16, coopmat, coopmat2, f16acc);
        }

#if defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
        if (!coopmat && !coopmat2 && matmul_id_type == MatMulIdType::NONE && is_legacy_quant(tname)) {
            string_to_spv(shader_name + "_" + tname + "_q8_1", "mul_mmq.comp", merge_maps(base_dict, {{"FLOAT_TYPE", FLOAT_TYPE(tname)}, {data_a_key, "1"}, {"D_TYPE", "float"},}), fp16, coopmat, coopmat2, f16acc);
        }
#endif
    }
}

void process_shaders() {
    std::cout << "ggml_vulkan: Generating and compiling shaders to SPIR-V" << std::endl;
    std::map<std::string, std::string> base_dict = {{"FLOAT_TYPE", "float"}};

    // matmul
    for (const MatMulIdType& matmul_id_type : {MatMulIdType::NONE, MatMulIdType::DEFAULT, MatMulIdType::SUBGROUP}) {
        // No coopmats
        // fp32
        matmul_shaders(false, matmul_id_type, false, false, false);

        // fp16, fp32acc and fp16acc
        matmul_shaders(true, matmul_id_type, false, false, false);
        matmul_shaders(true, matmul_id_type, false, false, true);

        if (matmul_id_type != MatMulIdType::DEFAULT) {
#if defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
            // Coopmat, fp32acc and fp16acc
            matmul_shaders(true, matmul_id_type, true, false, false);
            matmul_shaders(true, matmul_id_type, true, false, true);
#endif

#if defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
            // Coopmat2, fp32acc and fp16acc
            matmul_shaders(true, matmul_id_type, false, true, false);
            matmul_shaders(true, matmul_id_type, false, true, true);
#endif
        }
    }

    // flash attention
    for (const auto& f16acc : {false, true}) {
        std::map<std::string, std::string> fa_base_dict = base_dict;
        fa_base_dict["ACC_TYPE"] = f16acc ? "float16_t" : "float";
        fa_base_dict["ACC_TYPEV4"] = f16acc ? "f16vec4" : "vec4";
        if (f16acc) {
            fa_base_dict["ACC_TYPE_MAX"] = "\"float16_t(65504.0)\"";
        }

        for (const auto& tname : type_names) {
            if (tname == "f32") {
                continue;
            }
            if (tname == "bf16") continue;

#if defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
            if (tname == "f16") {
                string_to_spv("flash_attn_f32_f16_" + tname, "flash_attn_cm2.comp",
                    merge_maps(fa_base_dict, {{"Q_TYPE", "float"}, {"D_TYPE", "float"}}), true, false, true, f16acc);
            } else {
                std::string data_a_key = "DATA_A_" + to_uppercase(tname);
                string_to_spv("flash_attn_f32_f16_" + tname, "flash_attn_cm2.comp",
                    merge_maps(fa_base_dict, {{data_a_key, "1"}, {"Q_TYPE", "float"}, {"D_TYPE", "float"}, {"DEQUANTFUNC", "dequantFunc"+to_uppercase(tname) }, {"BLOCK_SIZE", "QUANT_K_"+to_uppercase(tname) }}), true, false, true, f16acc);
            }
#endif
#if defined(GGML_VULKAN_COOPMAT_GLSLC_SUPPORT)
            if (tname == "f16") {
                string_to_spv("flash_attn_f32_f16_" + tname, "flash_attn_cm1.comp",
                    merge_maps(fa_base_dict, {{"Q_TYPE", "float"}, {"D_TYPE", "float"}, {"COOPMAT", "1"}}), true, true, false, f16acc);
            } else if (tname == "q4_0" || tname == "q8_0") {
                std::string data_a_key = "DATA_A_" + to_uppercase(tname);
                string_to_spv("flash_attn_f32_f16_" + tname, "flash_attn_cm1.comp",
                    merge_maps(fa_base_dict, {{data_a_key, "1"}, {"Q_TYPE", "float"}, {"D_TYPE", "float"}, {"BLOCK_SIZE", "QUANT_K_"+to_uppercase(tname)}, {"COOPMAT", "1"}}), true, true, false, f16acc);
            }
#endif
            if (tname == "f16") {
                string_to_spv("flash_attn_f32_f16_" + tname, "flash_attn.comp",
                    merge_maps(fa_base_dict, {{"Q_TYPE", "float"}, {"D_TYPE", "float"}}), true, false, false, f16acc);
            } else if (tname == "q4_0" || tname == "q8_0") {
                std::string data_a_key = "DATA_A_" + to_uppercase(tname);
                string_to_spv("flash_attn_f32_f16_" + tname, "flash_attn.comp",
                    merge_maps(fa_base_dict, {{data_a_key, "1"}, {"Q_TYPE", "float"}, {"D_TYPE", "float"}, {"BLOCK_SIZE", "QUANT_K_"+to_uppercase(tname) }}), true, false, false, f16acc);
            }
        }
    }

    for (const auto& tname : type_names) {
        // mul mat vec
        std::string data_a_key = "DATA_A_" + to_uppercase(tname);
        std::string shader = (string_ends_with(tname, "_k") || string_starts_with(tname, "iq1_") || string_starts_with(tname, "iq2_") || string_starts_with(tname, "iq3_")) ? "mul_mat_vec_" + tname + ".comp" : "mul_mat_vec.comp";

        string_to_spv("mul_mat_vec_" + tname + "_f32_f32", shader, merge_maps(base_dict, {{data_a_key, "1"}, {"B_TYPE", "float"}, {"B_TYPE_VEC2", "vec2"}, {"B_TYPE_VEC4", "vec4"}, {"D_TYPE", "float"}}));
        string_to_spv("mul_mat_vec_" + tname + "_f16_f32", shader, merge_maps(base_dict, {{data_a_key, "1"}, {"B_TYPE", "float16_t"}, {"B_TYPE_VEC2", "f16vec2"}, {"B_TYPE_VEC4", "f16vec4"}, {"D_TYPE", "float"}}));

        string_to_spv("mul_mat_vec_" + tname + "_f32_f32_subgroup", shader, merge_maps(base_dict, {{data_a_key, "1"}, {"B_TYPE", "float"}, {"B_TYPE_VEC2", "vec2"}, {"B_TYPE_VEC4", "vec4"}, {"D_TYPE", "float"}, {"USE_SUBGROUP_ADD", "1"}}));
        string_to_spv("mul_mat_vec_" + tname + "_f16_f32_subgroup", shader, merge_maps(base_dict, {{data_a_key, "1"}, {"B_TYPE", "float16_t"}, {"B_TYPE_VEC2", "f16vec2"}, {"B_TYPE_VEC4", "f16vec4"}, {"D_TYPE", "float"}, {"USE_SUBGROUP_ADD", "1"}}));

        string_to_spv("mul_mat_vec_" + tname + "_f32_f32_subgroup_no_shmem", shader, merge_maps(base_dict, {{data_a_key, "1"}, {"B_TYPE", "float"}, {"B_TYPE_VEC2", "vec2"}, {"B_TYPE_VEC4", "vec4"}, {"D_TYPE", "float"}, {"USE_SUBGROUP_ADD_NO_SHMEM", "1"}}));
        string_to_spv("mul_mat_vec_" + tname + "_f16_f32_subgroup_no_shmem", shader, merge_maps(base_dict, {{data_a_key, "1"}, {"B_TYPE", "float16_t"}, {"B_TYPE_VEC2", "f16vec2"}, {"B_TYPE_VEC4", "f16vec4"}, {"D_TYPE", "float"}, {"USE_SUBGROUP_ADD_NO_SHMEM", "1"}}));

        string_to_spv("mul_mat_vec_id_" + tname + "_f32", shader, merge_maps(base_dict, {{"MUL_MAT_ID", "1"}, {data_a_key, "1"}, {"B_TYPE", "float"}, {"B_TYPE_VEC2", "vec2"}, {"B_TYPE_VEC4", "vec4"}, {"D_TYPE", "float"}}));

        // mul mat vec with integer dot product
#if defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
        if (is_legacy_quant(tname)) {
            string_to_spv("mul_mat_vec_" + tname + "_q8_1_f32", "mul_mat_vecq.comp", merge_maps(base_dict, {{data_a_key, "1"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}, {"FLOAT_TYPE_VEC2", "vec2"}, {"ACC_TYPE", "float"}}));
            string_to_spv("mul_mat_vec_" + tname + "_q8_1_f32_subgroup", "mul_mat_vecq.comp", merge_maps(base_dict, {{data_a_key, "1"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}, {"FLOAT_TYPE_VEC2", "vec2"}, {"ACC_TYPE", "float"}, {"USE_SUBGROUP_ADD", "1"}}));
            string_to_spv("mul_mat_vec_" + tname + "_q8_1_f32_subgroup_no_shmem", "mul_mat_vecq.comp", merge_maps(base_dict, {{data_a_key, "1"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}, {"FLOAT_TYPE_VEC2", "vec2"}, {"ACC_TYPE", "float"}, {"USE_SUBGROUP_ADD_NO_SHMEM", "1"}}));
        }
#endif

        // Dequant shaders
        if (tname != "f16" && tname != "bf16") {
            string_to_spv("dequant_" + tname, "dequant_" + tname + ".comp", merge_maps(base_dict, {{data_a_key, "1"}, {"D_TYPE", "float16_t"}}));
        }

        if (!string_ends_with(tname, "_k")) {
            shader = (tname == "f32" || tname == "f16" || tname == "bf16") ? "get_rows.comp" : "get_rows_quant.comp";

            if (tname == "f16") {
                string_to_spv("get_rows_" + tname, shader, merge_maps(base_dict, {{data_a_key, "1"}, {"B_TYPE", "int"}, {"D_TYPE", "float16_t"}, {"OPTIMIZATION_ERROR_WORKAROUND", "1"}}));
            } else {
                string_to_spv("get_rows_" + tname, shader, merge_maps(base_dict, {{data_a_key, "1"}, {"B_TYPE", "int"}, {"D_TYPE", "float16_t"}}));
            }
            string_to_spv("get_rows_" + tname + "_f32", shader, merge_maps(base_dict, {{data_a_key, "1"}, {"B_TYPE", "int"}, {"D_TYPE", "float"}}));
        }
    }

    string_to_spv("mul_mat_vec_p021_f16_f32_subgroup_add", "mul_mat_vec_p021.comp", {{"A_TYPE", "float16_t"}, {"A_TYPE_VEC4", "f16vec4"}, {"B_TYPE", "float"}, {"B_TYPE_VEC4", "vec4"}, {"D_TYPE", "float"}, {"USE_SUBGROUP_ADD", "1"}});
    string_to_spv("mul_mat_vec_p021_f16_f32",              "mul_mat_vec_p021.comp", {{"A_TYPE", "float16_t"}, {"A_TYPE_VEC4", "f16vec4"}, {"B_TYPE", "float"}, {"B_TYPE_VEC4", "vec4"}, {"D_TYPE", "float"}});
    string_to_spv("mul_mat_vec_nc_f16_f32", "mul_mat_vec_nc.comp", {{"A_TYPE", "float16_t"}, {"A_TYPE_VEC4", "f16vec4"}, {"B_TYPE", "float"}, {"B_TYPE_VEC4", "vec4"}, {"D_TYPE", "float"}});

    // Norms
    string_to_spv("norm_f32", "norm.comp", merge_maps(base_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}}));
    string_to_spv("group_norm_f32", "group_norm.comp", merge_maps(base_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}}));
    string_to_spv("rms_norm_f32", "rms_norm.comp", merge_maps(base_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}}));
    string_to_spv("rms_norm_partials_f32", "rms_norm_partials.comp", merge_maps(base_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}}));
    string_to_spv("rms_norm_back_f32", "rms_norm_back.comp", merge_maps(base_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}}));
    string_to_spv("l2_norm_f32", "l2_norm.comp", merge_maps(base_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}}));

    string_to_spv("cpy_f32_f32", "copy.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}});
    string_to_spv("cpy_f32_f16", "copy.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float16_t"}});
    string_to_spv("cpy_f16_f16", "copy.comp", {{"A_TYPE", "float16_t"}, {"D_TYPE", "float16_t"}, {"OPTIMIZATION_ERROR_WORKAROUND", "1"}});
    string_to_spv("cpy_f16_f32", "copy.comp", {{"A_TYPE", "float16_t"}, {"D_TYPE", "float"}, {"OPTIMIZATION_ERROR_WORKAROUND", "1"}});
    string_to_spv("cpy_f32_bf16","copy.comp", {{"A_TYPE", "float"}, {"D_TYPE", "uint16_t"}, {"DATA_D_BF16", "1"}});
    string_to_spv("contig_cpy_f32_f32", "contig_copy.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}});
    string_to_spv("contig_cpy_f32_i32", "contig_copy.comp", {{"A_TYPE", "float"}, {"D_TYPE", "int"}});
    string_to_spv("contig_cpy_i32_f32", "contig_copy.comp", {{"A_TYPE", "int"}, {"D_TYPE", "float"}});
    string_to_spv("contig_cpy_f32_f16", "contig_copy.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float16_t"}});
    string_to_spv("contig_cpy_f16_f16", "contig_copy.comp", {{"A_TYPE", "float16_t"}, {"D_TYPE", "float16_t"}, {"OPTIMIZATION_ERROR_WORKAROUND", "1"}});
    string_to_spv("contig_cpy_f16_f32", "contig_copy.comp", {{"A_TYPE", "float16_t"}, {"D_TYPE", "float"}, {"OPTIMIZATION_ERROR_WORKAROUND", "1"}});
    string_to_spv("contig_cpy_f32_bf16","contig_copy.comp",{{"A_TYPE", "float"}, {"D_TYPE", "uint16_t"}, {"DATA_D_BF16", "1"}});
    string_to_spv("cpy_f32_i32", "copy.comp", {{"A_TYPE", "float"}, {"D_TYPE", "int"}});
    string_to_spv("cpy_i32_f32", "copy.comp", {{"A_TYPE", "int"}, {"D_TYPE", "float"}});

    for (std::string t : {"q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "iq4_nl"}) {
        string_to_spv("cpy_f32_" + t, "copy_to_quant.comp", {{"DATA_A_" + to_uppercase(t), "1"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});
        string_to_spv("cpy_f32_" + t + "_rte", "copy_to_quant.comp", {{"DATA_A_" + to_uppercase(t), "1"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}, {"RTE16", "1"}});
        string_to_spv("cpy_" + t + "_f32", "copy_from_quant.comp", {{"DATA_A_" + to_uppercase(t), "1"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});
    }

    for (std::string t : {"f32", "f16", "bf16", "q4_0", "q4_1", "q5_0", "q5_1", "q8_0", "iq4_nl"}) {
        string_to_spv("set_rows_" + t, "copy_to_quant.comp", {{"SET_ROWS", "1"}, {"DATA_A_" + to_uppercase(t), "1"}, {"B_TYPE", "uvec2"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});
        string_to_spv("set_rows_" + t + "_rte", "copy_to_quant.comp", {{"SET_ROWS", "1"}, {"DATA_A_" + to_uppercase(t), "1"}, {"B_TYPE", "uvec2"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}, {"RTE16", "1"}});
    }

    auto get_type_str = [](bool f16) {
        return f16 ? "float16_t" : "float";
    };
    auto get_suffix = [](bool src0_f16, bool src1_f16, bool dst_f16) {
        std::string s;
        s += std::string(src0_f16 ? "_f16" : "_f32");
        s += std::string(src1_f16 ? "_f16" : "_f32");
        s += std::string(dst_f16 ? "_f16" : "_f32");
        return s;
    };
    for (std::string op : {"add", "sub", "mul", "div", "add_rms", }) {
    for (auto src0_f16 : {false, true}) {
    for (auto src1_f16 : {false, true}) {
    for (auto dst_f16  : {false, true}) {
    for (auto rte      : {false, true}) {
        auto source = op == "add_rms" ? std::string("add") : op;
        auto name = op + get_suffix(src0_f16, src1_f16, dst_f16) + (rte ? "_rte" : "");
        auto add_rms = op == "add_rms" ? "1" : "0";
        string_to_spv(name.c_str(), source + ".comp", {{"A_TYPE", get_type_str(src0_f16)}, {"B_TYPE", get_type_str(src1_f16)}, {"D_TYPE", get_type_str(dst_f16)}, {"FLOAT_TYPE", "float"}, {"RTE16", rte ? "1" : "0"}, {"ADD_RMS" , add_rms}});
    }
    }
    }
    }
    }

    string_to_spv("sub_f32", "sub.comp", {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});

    string_to_spv("acc_f32", "acc.comp", {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});

    string_to_spv("split_k_reduce", "mul_mat_split_k_reduce.comp", {});
    string_to_spv("fa_split_k_reduce", "flash_attn_split_k_reduce.comp", {});

    string_to_spv("quantize_q8_1", "quantize_q8_1.comp", {});
    string_to_spv("quantize_q8_1_subgroup", "quantize_q8_1.comp", {{"USE_SUBGROUPS", "1"}});

    string_to_spv("quantize_q8_1_x4", "quantize_q8_1.comp", {{"QBLOCK_X4", "1"}});
    string_to_spv("quantize_q8_1_x4_subgroup", "quantize_q8_1.comp", {{"QBLOCK_X4", "1"}, {"USE_SUBGROUPS", "1"}});

    string_to_spv("mul_f32", "mul.comp", {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});

    string_to_spv("div_f32", "div.comp", {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});

    string_to_spv("repeat_f32", "repeat.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}});
    string_to_spv("repeat_back_f32", "repeat_back.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}});

    string_to_spv("scale_f32", "scale.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});

    string_to_spv("sqr_f32", "square.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});

    string_to_spv("sqrt_f32", "sqrt.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});

    string_to_spv("sin_f32", "sin.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});

    string_to_spv("cos_f32", "cos.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});

    string_to_spv("clamp_f32", "clamp.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}});

    string_to_spv("pad_f32", "pad.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}});

    string_to_spv("concat_f32", "concat.comp", {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}});
    string_to_spv("concat_f16", "concat.comp", {{"A_TYPE", "float16_t"}, {"B_TYPE", "float16_t"}, {"D_TYPE", "float16_t"}, {"OPTIMIZATION_ERROR_WORKAROUND", "1"}});
    string_to_spv("concat_i32", "concat.comp", {{"A_TYPE", "int"}, {"B_TYPE", "int"}, {"D_TYPE", "int"}});

    string_to_spv("upscale_f32", "upscale.comp", {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}});

    string_to_spv("exp_f16",        "exp.comp",         {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}});
    string_to_spv("exp_f32",        "exp.comp",         {{"A_TYPE", "float"},       {"D_TYPE", "float"}});
    string_to_spv("gelu_f16",       "gelu.comp",        {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}});
    string_to_spv("gelu_f32",       "gelu.comp",        {{"A_TYPE", "float"},       {"D_TYPE", "float"}});
    string_to_spv("gelu_erf_f16",   "gelu_erf.comp",    {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}});
    string_to_spv("gelu_erf_f32",   "gelu_erf.comp",    {{"A_TYPE", "float"},       {"D_TYPE", "float"}});
    string_to_spv("gelu_quick_f16", "gelu_quick.comp",  {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}});
    string_to_spv("gelu_quick_f32", "gelu_quick.comp",  {{"A_TYPE", "float"},       {"D_TYPE", "float"}});
    string_to_spv("silu_f16",       "silu.comp",        {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}});
    string_to_spv("silu_f32",       "silu.comp",        {{"A_TYPE", "float"},       {"D_TYPE", "float"}});
    string_to_spv("relu_f16",       "relu.comp",        {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}});
    string_to_spv("relu_f32",       "relu.comp",        {{"A_TYPE", "float"},       {"D_TYPE", "float"}});
    string_to_spv("tanh_f16",       "tanh.comp",        {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}});
    string_to_spv("tanh_f32",       "tanh.comp",        {{"A_TYPE", "float"},       {"D_TYPE", "float"}});
    string_to_spv("sigmoid_f16",    "sigmoid.comp",     {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}});
    string_to_spv("sigmoid_f32",    "sigmoid.comp",     {{"A_TYPE", "float"},       {"D_TYPE", "float"}});
    string_to_spv("hardsigmoid_f16","hardsigmoid.comp", {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}});
    string_to_spv("hardsigmoid_f32","hardsigmoid.comp", {{"A_TYPE", "float"},       {"D_TYPE", "float"}});
    string_to_spv("hardswish_f16",  "hardswish.comp",   {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"}});
    string_to_spv("hardswish_f32",  "hardswish.comp",   {{"A_TYPE", "float"},       {"D_TYPE", "float"}});

    for (auto rte : {false, true}) {
        std::string suffix = rte ? "_rte" : "";
        string_to_spv("geglu_f16" + suffix,      "geglu.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"},   {"RTE16", rte ? "1" : "0"}});
        string_to_spv("geglu_f32" + suffix,      "geglu.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},       {"RTE16", rte ? "1" : "0"}});
        string_to_spv("reglu_f16" + suffix,      "reglu.comp",       {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"},   {"RTE16", rte ? "1" : "0"}});
        string_to_spv("reglu_f32" + suffix,      "reglu.comp",       {{"A_TYPE", "float"},       {"D_TYPE", "float"},       {"RTE16", rte ? "1" : "0"}});
        string_to_spv("swiglu_f16" + suffix,     "swiglu.comp",      {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"},   {"RTE16", rte ? "1" : "0"}});
        string_to_spv("swiglu_f32" + suffix,     "swiglu.comp",      {{"A_TYPE", "float"},       {"D_TYPE", "float"},       {"RTE16", rte ? "1" : "0"}});
        string_to_spv("swiglu_oai_f16" + suffix, "swiglu_oai.comp",  {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"},   {"RTE16", rte ? "1" : "0"}});
        string_to_spv("swiglu_oai_f32" + suffix, "swiglu_oai.comp",  {{"A_TYPE", "float"},       {"D_TYPE", "float"},       {"RTE16", rte ? "1" : "0"}});
        string_to_spv("geglu_erf_f16" + suffix,  "geglu_erf.comp",   {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"},   {"RTE16", rte ? "1" : "0"}});
        string_to_spv("geglu_erf_f32" + suffix,  "geglu_erf.comp",   {{"A_TYPE", "float"},       {"D_TYPE", "float"},       {"RTE16", rte ? "1" : "0"}});
        string_to_spv("geglu_quick_f16" + suffix,"geglu_quick.comp", {{"A_TYPE", "float16_t"},   {"D_TYPE", "float16_t"},   {"RTE16", rte ? "1" : "0"}});
        string_to_spv("geglu_quick_f32" + suffix,"geglu_quick.comp", {{"A_TYPE", "float"},       {"D_TYPE", "float"},       {"RTE16", rte ? "1" : "0"}});
    }

    string_to_spv("leaky_relu_f32", "leaky_relu.comp",  {{"A_TYPE", "float"}, {"D_TYPE", "float"}});
    string_to_spv("silu_back_f32",  "silu_back.comp",   {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}});

    string_to_spv("diag_mask_inf_f32", "diag_mask_inf.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}});

    string_to_spv("soft_max_f32", "soft_max.comp", merge_maps(base_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}}));
    string_to_spv("soft_max_f32_f16", "soft_max.comp", merge_maps(base_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float16_t"}, {"D_TYPE", "float"}}));
    string_to_spv("soft_max_back_f32", "soft_max_back.comp", merge_maps(base_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}}));

    string_to_spv("rope_norm_f32", "rope_norm.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}});
    string_to_spv("rope_norm_f16", "rope_norm.comp", {{"A_TYPE", "float16_t"}, {"D_TYPE", "float16_t"}});
    string_to_spv("rope_norm_f16_rte", "rope_norm.comp", {{"A_TYPE", "float16_t"}, {"D_TYPE", "float16_t"}, {"RTE16", "1"}});

    string_to_spv("rope_neox_f32", "rope_neox.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}});
    string_to_spv("rope_neox_f16", "rope_neox.comp", {{"A_TYPE", "float16_t"}, {"D_TYPE", "float16_t"}});
    string_to_spv("rope_neox_f16_rte", "rope_neox.comp", {{"A_TYPE", "float16_t"}, {"D_TYPE", "float16_t"}, {"RTE16", "1"}});

    string_to_spv("rope_multi_f32", "rope_multi.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}});
    string_to_spv("rope_multi_f16", "rope_multi.comp", {{"A_TYPE", "float16_t"}, {"D_TYPE", "float16_t"}});
    string_to_spv("rope_multi_f16_rte", "rope_multi.comp", {{"A_TYPE", "float16_t"}, {"D_TYPE", "float16_t"}, {"RTE16", "1"}});

    string_to_spv("rope_vision_f32", "rope_vision.comp", {{"A_TYPE", "float"}, {"D_TYPE", "float"}});
    string_to_spv("rope_vision_f16", "rope_vision.comp", {{"A_TYPE", "float16_t"}, {"D_TYPE", "float16_t"}});
    string_to_spv("rope_vision_f16_rte", "rope_vision.comp", {{"A_TYPE", "float16_t"}, {"D_TYPE", "float16_t"}, {"RTE16", "1"}});

    string_to_spv("argsort_f32", "argsort.comp", {{"A_TYPE", "float"}});

    string_to_spv("argmax_f32", "argmax.comp", merge_maps(base_dict, {{"A_TYPE", "float"}, {"D_TYPE", "int"}}));
    string_to_spv("sum_rows_f32", "sum_rows.comp", merge_maps(base_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}}));
    string_to_spv("count_equal_i32", "count_equal.comp", merge_maps(base_dict, {{"A_TYPE", "int"}, {"B_TYPE", "int"}, {"D_TYPE", "int"}}));

    string_to_spv("im2col_f32", "im2col.comp", merge_maps(base_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}}));
    string_to_spv("im2col_f32_f16", "im2col.comp", merge_maps(base_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float16_t"}}));
    string_to_spv("im2col_f32_f16_rte", "im2col.comp", merge_maps(base_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float16_t"}, {"RTE16", "1"}}));

    string_to_spv("im2col_3d_f32", "im2col_3d.comp", merge_maps(base_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}}));
    string_to_spv("im2col_3d_f32_f16", "im2col_3d.comp", merge_maps(base_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float16_t"}}));
    string_to_spv("im2col_3d_f32_f16_rte", "im2col_3d.comp", merge_maps(base_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float16_t"}, {"RTE16", "1"}}));

    string_to_spv("timestep_embedding_f32", "timestep_embedding.comp", merge_maps(base_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}}));

    string_to_spv("conv_transpose_1d_f32", "conv_transpose_1d.comp", {{"A_TYPE", "float"},  {"B_TYPE", "float"}, {"D_TYPE", "float"}});

    string_to_spv("pool2d_f32", "pool2d.comp", merge_maps(base_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}}));

    string_to_spv("rwkv_wkv6_f32", "wkv6.comp", merge_maps(base_dict, {{"A_TYPE", "float"}}));

    string_to_spv("rwkv_wkv7_f32", "wkv7.comp", merge_maps(base_dict, {{"A_TYPE", "float"}}));

    string_to_spv("opt_step_adamw_f32", "opt_step_adamw.comp", merge_maps(base_dict, {{"A_TYPE", "float"}}));
    string_to_spv("opt_step_sgd_f32", "opt_step_sgd.comp", merge_maps(base_dict, {{"A_TYPE", "float"}}));

    string_to_spv("conv2d_f32_unroll", "conv2d_mm.comp", {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"USE_COLLECTIVES", "1"}, {"UNROLL", "[[unroll]]"}});
    string_to_spv("conv2d_f16_f32_unroll", "conv2d_mm.comp", {{"A_TYPE", "float16_t"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"USE_COLLECTIVES", "1"}, {"UNROLL", "[[unroll]]"}});

    string_to_spv("conv2d_f32", "conv2d_mm.comp", {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"USE_COLLECTIVES", "1"}, {"UNROLL", ""}});
    string_to_spv("conv2d_f16_f32", "conv2d_mm.comp", {{"A_TYPE", "float16_t"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"USE_COLLECTIVES", "1"}, {"UNROLL", ""}});

#if defined(GGML_VULKAN_COOPMAT2_GLSLC_SUPPORT)
    string_to_spv("conv2d_f32", "conv2d_mm.comp", {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"USE_COLLECTIVES", "1"}, {"UNROLL", "[[unroll]]"}, {"COOPMAT2", "1"}}, true, false, true);
    string_to_spv("conv2d_f16_f32", "conv2d_mm.comp", {{"A_TYPE", "float16_t"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"USE_COLLECTIVES", "1"}, {"UNROLL", "[[unroll]]"}, {"COOPMAT2", "1"}}, true, false, true);
#endif

    string_to_spv("conv2d_dw_whcn_f32", "conv2d_dw.comp", merge_maps(base_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"WHCN", "1"}}));
    string_to_spv("conv2d_dw_cwhn_f32", "conv2d_dw.comp", merge_maps(base_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"CWHN", "1"}}));
    string_to_spv("conv2d_dw_whcn_f16_f32", "conv2d_dw.comp", merge_maps(base_dict, {{"A_TYPE", "float16_t"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"WHCN", "1"}}));
    string_to_spv("conv2d_dw_cwhn_f16_f32", "conv2d_dw.comp", merge_maps(base_dict, {{"A_TYPE", "float16_t"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"CWHN", "1"}}));

    string_to_spv("roll_f32", "roll.comp", merge_maps(base_dict, {{"A_TYPE", "float"}, {"D_TYPE", "float"}}));

    string_to_spv("add_id_f32", "add_id.comp", merge_maps(base_dict, {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}}));

    string_to_spv("multi_add_f32", "multi_add.comp", {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}, {"RTE16", "1"}, {"ADD_RMS" , "0"}});
    string_to_spv("multi_add_rms_f32", "multi_add.comp", {{"A_TYPE", "float"}, {"B_TYPE", "float"}, {"D_TYPE", "float"}, {"FLOAT_TYPE", "float"}, {"RTE16", "1"}, {"ADD_RMS" , "1"}});
}


void write_embed_files(const path& target_hpp, const path& target_cpp, bool no_embed) {
    std::stringstream hdr = make_generic_stringstream();
    std::stringstream src = make_generic_stringstream();

    hdr << "#include <cstdint>\n\n";
    src << "#include \"" << target_hpp.filename().string() << "\"\n\n";

    if (no_embed) {
        hdr << "#define GGML_VK_SHADER_DIR \"" << output_dir.generic_string() << "\"\n\n";
    }

    std::sort(shader_fnames.begin(), shader_fnames.end());
    for (const auto& pair : shader_fnames) {
        auto && [name, path] = pair;
        if (no_embed) {
            hdr << "inline constexpr char const * " << name << "_data = \"" << path.filename().string() << "\";\n";
            hdr << "const uint64_t " << name << "_len = 0;\n\n";
        } else {
            std::vector<unsigned char> data = read_binary_file(path);
            if (data.empty()) {
                continue;
            }

            hdr << "extern const unsigned char " << name << "_data[" << data.size() << "];\n";
            hdr << "const uint64_t " << name << "_len = " << data.size() << ";\n\n";

            src << "const unsigned char " << name << "_data[" << data.size() << "] = {\n" << std::hex;
            for (size_t i = 0; i < data.size(); ++i) {
                src << "0x" << static_cast<int>(data[i]) << ",";
                if ((i + 1) % 12 == 0) src << "\n";
            }
            src << std::dec << "\n};\n\n";
        }
    }

    std::string suffixes[2] = {"_f32", "_f16"};
    for (const char *op : {"add", "sub", "mul", "div", "add_rms"}) {
        hdr << "extern const void * " << op << "_data[2][2][2][2];\n";
        hdr << "extern const uint64_t " << op << "_len[2][2][2][2];\n";

        std::stringstream data = make_generic_stringstream();
        std::stringstream len  = make_generic_stringstream();
        data << "const void * " << op << "_data[2][2][2][2] = ";
        len  << "const uint64_t " << op << "_len[2][2][2][2] = ";
        for (uint32_t t0 = 0; t0 < 2; ++t0) {
            if (t0 == 0) {
                data << "{";
                len  << "{";
            }
            for (uint32_t t1 = 0; t1 < 2; ++t1) {
                if (t1 == 0) {
                    data << "{";
                    len  << "{";
                }
                for (uint32_t t2 = 0; t2 < 2; ++t2) {
                    if (t2 == 0) {
                        data << "{";
                        len  << "{";
                    }
                    for (uint32_t rte = 0; rte < 2; ++rte) {
                        if (rte == 0) {
                            data << "{";
                            len  << "{";
                        }
                        data << op << suffixes[t0] << suffixes[t1] << suffixes[t2] << ((rte != 0) ? "_rte" : "");
                        len  << op << suffixes[t0] << suffixes[t1] << suffixes[t2] << ((rte != 0) ? "_rte" : "");
                        data << "_data,";
                        len  << "_len,";
                        if (rte == 1) {
                            data << "}, ";
                            len  << "}, ";
                        }
                    }
                    if (t2 == 1) {
                        data << "}, ";
                        len  << "}, ";
                    }
                }
                if (t1 == 1) {
                    data << "}, ";
                    len  << "}, ";
                }
            }
            if (t0 == 1) {
                data << "};\n";
                len  << "};\n";
            }
        }
        src << data.str();
        src << len.str();
    }

    std::vector<std::string> btypes = {"f16", "f32"};

#if defined(GGML_VULKAN_INTEGER_DOT_GLSLC_SUPPORT)
    btypes.push_back("q8_1");
#endif

    for (const std::string& btype : btypes) {
    for (const auto& tname : type_names) {
        if (btype == "q8_1" && !is_legacy_quant(tname)) {
            continue;
        }
        hdr << "extern const void * arr_dmmv_"   << tname << "_" << btype << "_f32_data[3];\n";
        hdr << "extern const uint64_t arr_dmmv_" << tname << "_" << btype << "_f32_len[3];\n";
        src << "const void * arr_dmmv_"   << tname << "_" << btype << "_f32_data[3] = {mul_mat_vec_" << tname << "_" << btype << "_f32_data, mul_mat_vec_" << tname << "_" << btype << "_f32_subgroup_data, mul_mat_vec_" << tname << "_" << btype << "_f32_subgroup_no_shmem_data};\n";
        src << "const uint64_t arr_dmmv_" << tname << "_" << btype << "_f32_len[3] =  {mul_mat_vec_" << tname << "_" << btype << "_f32_len,  mul_mat_vec_" << tname << "_" << btype << "_f32_subgroup_len, mul_mat_vec_"  << tname << "_" << btype << "_f32_subgroup_no_shmem_len};\n";
    }
    }

    write_file_if_changed(target_hpp, hdr.str());
    if (no_embed) {
        write_file_if_changed(target_cpp, src.str());
    } else {
        write_binary_file(target_cpp, src.str());
    }
}

} // namespace

int main(int argc, char** argv) {
    std::map<std::string, std::string> args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.rfind("--", 0) == 0) {
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                args[arg] = argv[i + 1];
                ++i;
            } else {
                args[arg] = "";
            }
        }
    }

    path target_hpp = "ggml-vulkan-shaders.hpp";
    path target_cpp = "ggml-vulkan-shaders.cpp";
    path target_cmake;
    bool no_embed = false;

    if (args.find("--glslc") != args.end()) {
        GLSLC = args["--glslc"]; // Path to glslc
    }
    if (args.find("--input-dir") != args.end()) {
        input_dir = args["--input-dir"]; // Directory containing shader sources
    }
    if (args.find("--output-dir") != args.end()) {
        output_dir = args["--output-dir"]; // Directory for containing SPIR-V output
    }
    if (args.find("--target-hpp") != args.end()) {
        target_hpp = args["--target-hpp"]; // Path to generated header file
    }
    if (args.find("--target-cpp") != args.end()) {
        target_cpp = args["--target-cpp"]; // Path to generated cpp file
    }
    if (args.find("--target-cmake") != args.end()) {
        target_cmake = args["--target-cmake"]; // Path to the generated CMakeLists.txt file
    }
    if (args.find("--no-embed") != args.end()) {
        no_embed = true; // Do not embed SPIR-V binaries into C++ source files, only write stubs to header
    }
    if (args.find("--help") != args.end()) {
        std::cout << usage << std::endl;
        return EXIT_SUCCESS;
    }

    if (no_embed && target_cmake.empty()) {
        std::cerr << "--no-embed requires --target-cmake to be specified\n";
        return EXIT_FAILURE;
    }

    try {
        if (!exists(input_dir)) {
            std::cerr << "Input directory does not exist: " << input_dir << "\n";
            return EXIT_FAILURE;
        }
        if (!exists(output_dir)) {
            create_directories(output_dir);
        }

        if (!target_cmake.empty()) {
            if (target_cmake.has_parent_path() && !exists(target_cmake.parent_path())) {
                create_directories(target_cmake.parent_path());
            }
            cmake.add_header(argc, argv);
        }

        process_shaders();

        if (target_cmake.empty() || no_embed) {
            write_embed_files(target_hpp, target_cpp, no_embed);
        }

        if (!target_cmake.empty()) {
            if (no_embed) {
                cmake.add_target_build_only();
            } else {
                cmake.add_target_embed(path(argv[0]), target_hpp, target_cpp);
            }
            cmake.write(target_cmake);
        }

    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
