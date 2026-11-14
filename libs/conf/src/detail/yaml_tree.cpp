// CONF — src/detail/yaml_tree.cpp
// Implementation of conf::detail::Yaml_Tree (private yaml-cpp wrapper).

#include "detail/yaml_tree.hpp"

#include <charconv>
#include <fstream>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

#include "conf/value.hpp"

namespace conf::detail {

// ── Private constructor ──────────────────────────────────────────────────────

Yaml_Tree::Yaml_Tree(YAML::Node root) noexcept : root_(std::move(root)) {}

// ── Move semantics ───────────────────────────────────────────────────────────

Yaml_Tree::Yaml_Tree(Yaml_Tree &&other) noexcept : root_(other.root_) {
    // yaml-cpp Node has reference semantics — assigning a new node to other.root_
    // would mutate the shared underlying data and corrupt our root_. Instead, we
    // leave the moved-from instance's root_ sharing the same data. This is safe
    // because the moved-from instance will either be destroyed (no-op) or
    // overwritten by another assignment. The key invariant is: we never call reset()
    // or assign to other.root_ here.
}

Yaml_Tree &Yaml_Tree::operator=(Yaml_Tree &&other) noexcept {
    if (this != &other) {
        root_ = other.root_;
        // Same: do NOT invalidate other.root_ due to yaml-cpp's reference semantics.
    }
    return *this;
}

// ── Factory: from_file ───────────────────────────────────────────────────────

Yaml_Tree Yaml_Tree::from_file(const std::string &path) {
    // Use std::ifstream to distinguish File_Not_Found from Parse_Error:
    // if the file cannot be opened at all, it's File_Not_Found.
    std::ifstream ifs(path);
    if (!ifs.is_open()) {
        throw Conf_Error(Error_Code::File_Not_Found, "cannot open file: " + path);
    }

    // Read entire file content, then parse via YAML::Load.
    std::ostringstream buf;
    buf << ifs.rdbuf();
    std::string content = buf.str();

    try {
        YAML::Node root = YAML::Load(content);
        return Yaml_Tree(root);
    } catch (const YAML::ParserException &e) {
        throw Conf_Error(Error_Code::Parse_Error, e.what());
    }
}

// ── Factory: from_string ─────────────────────────────────────────────────────

Yaml_Tree Yaml_Tree::from_string(const std::string &text) {
    try {
        YAML::Node root = YAML::Load(text);
        return Yaml_Tree(root);
    } catch (const YAML::ParserException &e) {
        throw Conf_Error(Error_Code::Parse_Error, e.what());
    }
}

// ── Dotted-path resolution ───────────────────────────────────────────────────

YAML::Node Yaml_Tree::resolve(std::string_view dotted_path) const {
    // Precondition: reject empty path
    if (dotted_path.empty()) {
        throw Conf_Error(Error_Code::Invalid_Arg, "empty path");
    }

    // Split on '.' and validate all segments are non-empty
    std::vector<std::string_view> segments;
    std::size_t start = 0;
    while (start <= dotted_path.size()) {
        auto dot = dotted_path.find('.', start);
        if (dot == std::string_view::npos) {
            segments.push_back(dotted_path.substr(start));
            break;
        }
        segments.push_back(dotted_path.substr(start, dot - start));
        start = dot + 1;
        // If dot is the last character, there's a trailing empty segment
        if (start == dotted_path.size()) {
            segments.push_back(std::string_view{});
            break;
        }
    }

    for (const auto &seg : segments) {
        if (seg.empty()) {
            throw Conf_Error(Error_Code::Invalid_Arg, "empty path segment");
        }
    }

    // Traverse read-only handles, then copy only the selected subtree.
    // Node assignment mutates shared YAML storage; reset merely rebinds this
    // local handle. Const child access avoids inserting missing keys.
    YAML::Node current(root_);

    for (const auto &seg : segments) {
        if (current.IsMap()) {
            // Look up segment by literal key (byte-for-byte, no trimming)
            std::string key_str(seg);
            YAML::Node child = static_cast<const YAML::Node &>(current)[key_str];
            if (!child.IsDefined()) {
                throw Conf_Error(Error_Code::Key_Not_Found, "key not found: " + std::string(dotted_path));
            }
            current.reset(child);

        } else if (current.IsSequence()) {
            // Check if segment is all ASCII digits
            bool all_digits = true;
            for (char c : seg) {
                if (c < '0' || c > '9') {
                    all_digits = false;
                    break;
                }
            }

            if (!all_digits) {
                throw Conf_Error(Error_Code::Key_Not_Found, "key not found: " + std::string(dotted_path));
            }

            // Safe index parsing: reject oversized digit strings and out-of-range
            // Use std::from_chars for safe, locale-independent parsing
            std::size_t index = 0;
            auto [ptr, ec] = std::from_chars(seg.data(), seg.data() + seg.size(), index);

            if (ec != std::errc{} || ptr != seg.data() + seg.size()) {
                // Overflow or parse failure (e.g., absurdly large number)
                throw Conf_Error(Error_Code::Key_Not_Found, "key not found: " + std::string(dotted_path));
            }

            // Const indexed lookup checks the underlying sequence directly;
            // avoid size(), whose yaml-cpp cache is lazily mutable.
            YAML::Node child = static_cast<const YAML::Node &>(current)[index];
            if (!child.IsDefined()) {
                throw Conf_Error(Error_Code::Key_Not_Found, "key not found: " + std::string(dotted_path));
            }
            current.reset(child);

        } else {
            // Scalar, Null, or Undefined — cannot descend further
            throw Conf_Error(Error_Code::Key_Not_Found, "key not found: " + std::string(dotted_path));
        }
    }

    // Keep the original isolated-snapshot contract, including aliases and
    // mutable list access in callers. Unrelated document entries are not copied.
    return YAML::Clone(current);
}

// ── Node kind mapping ────────────────────────────────────────────────────────

Node_Kind Yaml_Tree::node_kind(const YAML::Node &node) noexcept {
    switch (node.Type()) {
        case YAML::NodeType::Undefined:
            return Node_Kind::Undefined;
        case YAML::NodeType::Null:
            return Node_Kind::Null;
        case YAML::NodeType::Scalar:
            return Node_Kind::Scalar;
        case YAML::NodeType::Sequence:
            return Node_Kind::Sequence;
        case YAML::NodeType::Map:
            return Node_Kind::Map;
        default:
            return Node_Kind::Undefined;
    }
}

// ── Typed conversion ─────────────────────────────────────────────────────────

template <>
int Yaml_Tree::convert<int>(const YAML::Node &node) const {
    if (!node.IsScalar()) {
        throw Conf_Error(Error_Code::Type_Mismatch, "node is not a scalar");
    }
    try {
        return node.as<int>();
    } catch (const YAML::BadConversion &) {
        throw Conf_Error(Error_Code::Type_Mismatch, "cannot convert scalar to int");
    }
}

template <>
double Yaml_Tree::convert<double>(const YAML::Node &node) const {
    if (!node.IsScalar()) {
        throw Conf_Error(Error_Code::Type_Mismatch, "node is not a scalar");
    }
    try {
        return node.as<double>();
    } catch (const YAML::BadConversion &) {
        throw Conf_Error(Error_Code::Type_Mismatch, "cannot convert scalar to double");
    }
}

template <>
bool Yaml_Tree::convert<bool>(const YAML::Node &node) const {
    if (!node.IsScalar()) {
        throw Conf_Error(Error_Code::Type_Mismatch, "node is not a scalar");
    }
    try {
        return node.as<bool>();
    } catch (const YAML::BadConversion &) {
        throw Conf_Error(Error_Code::Type_Mismatch, "cannot convert scalar to bool");
    }
}

template <>
std::string Yaml_Tree::convert<std::string>(const YAML::Node &node) const {
    if (!node.IsScalar()) {
        throw Conf_Error(Error_Code::Type_Mismatch, "node is not a scalar");
    }
    try {
        return node.as<std::string>();
    } catch (const YAML::BadConversion &) {
        throw Conf_Error(Error_Code::Type_Mismatch, "cannot convert node to string");
    }
}

}  // namespace conf::detail
