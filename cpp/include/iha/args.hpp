#pragma once

#include <map>
#include <stdexcept>
#include <string>

namespace iha {

// "--anahtar deger" ve "--bayrak" bicimli basit komut satiri ayristirici.
class Args {
public:
    Args(int argc, char** argv) {
        for (int i = 1; i < argc; ++i) {
            std::string k = argv[i];
            if (k.rfind("--", 0) != 0) throw std::invalid_argument("beklenmeyen arguman: " + k);
            k = k.substr(2);
            if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0)
                kv_[k] = argv[++i];
            else
                kv_[k] = "1";
        }
    }
    bool has(const std::string& k) const { return kv_.count(k) > 0; }
    std::string str(const std::string& k, const std::string& def = "") const {
        auto it = kv_.find(k);
        return it == kv_.end() ? def : it->second;
    }
    std::string required(const std::string& k) const {
        if (!has(k)) throw std::invalid_argument("--" + k + " gerekli");
        return kv_.at(k);
    }
    float f(const std::string& k, float def) const { return has(k) ? std::stof(kv_.at(k)) : def; }
    int i(const std::string& k, int def) const { return has(k) ? std::stoi(kv_.at(k)) : def; }

private:
    std::map<std::string, std::string> kv_;
};

}  // namespace iha
