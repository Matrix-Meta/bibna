cmake_minimum_required(VERSION 3.10)
project(BrainBitNet)

find_package(IntelDPCPP REQUIRED)

add_executable(brain_bitnet main.cpp)
# 如果是單一檔案編譯，只需 main.cpp，因為 header only 實作
# 確保包含 src 目錄
target_include_directories(brain_bitnet PRIVATE ./src)

# 連結 SYCL
target_link_libraries(brain_bitnet PRIVATE SYCL::SYCL)
