#include <iostream>

#include "scope/system.hpp"
#include "scope/init.hpp"

int main(int argc, char **argv) {

    using TransferMethod = scope::system::TransferMethod;

    scope::initialize(&argc, argv);

    std::vector<MemorySpace> spaces = scope::system::memory_spaces();

    for (const auto &space : spaces) {
        std::cout << space << "\n";
    }

    for (size_t i = 0; i < spaces.size(); ++i) {
        for (size_t j = 0; j < spaces.size(); ++j) {
            std::cout << spaces[i] << " -> " << spaces[j] << "\n";
            std::vector<TransferMethod> methods = scope::system::transfer_methods(spaces[i], spaces[j]);
            for (const auto &method : methods) {
                std::cout << "\t" << method << "\n";
            }
        }
    }


}