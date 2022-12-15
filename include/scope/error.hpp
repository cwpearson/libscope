#pragma once

#include <iostream>
#include <string>

#include "scope/logger.hpp"

template <typename T>
const char *error_string(const T &error);

#if defined(SCOPE_USE_CUDA)
template <> inline const char *error_string<cudaError_t>(const cudaError_t &status) {
  return cudaGetErrorString(status);
}
#endif

#if defined(SCOPE_USE_HIP)
template <> inline const char *error_string<hipError_t>(const hipError_t &status) {
  return hipGetErrorString(status);
}
#endif

template <typename T>
bool is_success(const T &err);

#if defined(SCOPE_USE_CUDA)
template <> inline bool is_success<cudaError_t>(const cudaError_t &err) {
  return err == cudaSuccess;
}
#endif

#if defined(SCOPE_USE_HIP)
template <> inline bool is_success<hipError_t>(const hipError_t &err) {
  return err == hipSuccess;
}
#endif

template <typename T>
bool is_error(const T &err) {
  return !is_success<T>(err);
}

  template <typename T>
  bool print_if_error(const T &err, const char *stmt, const char *file, const char *func,
                                           int line) {
    if (is_success<T>(err)) {
      return false;
    }
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    // in device code
    printf("ERROR on %s::%d In %s:(%s) FAILED\n", file, line, func, stmt);
#else  // defined(__CUDA_ARCH__)
    // in host code
    LOG(critical, "ERROR[{}] on {}:{} In {}:({}) FAILED", error_string<T>(err), file, line, func, stmt);
#endif // defined(__CUDA_ARCH__)
    return true;
  }

  template <typename T>
  inline bool throw_if_error(const T err, const char *msg, const char *stmt, const char *file,
                                           const char *func, int line) {
    if (is_success<T>(err)) {
      return false;
    }
#if defined(__CUDA_ARCH__)
    // in device code
    printf("ERROR on %s::%d In %s:(%s) FAILED\n", file, line, func, stmt);
#else  // defined(__CUDA_ARCH__)
    // in host code
    if (msg == nullptr) {
      LOG(critical, "ERROR[{}] on {}::{} In {}:({}) FAILED", error_string<T>(err), file, line, func, stmt);
      throw std::runtime_error(
          fmt::format("ERROR[{}] on {}::{} In {}:({}) FAILED", error_string<T>(err), file, line, func, stmt));
    }
    LOG(critical, "ERROR[{}] because of {} on {}::{} In {}:({}) FAILED", error_string<T>(err), msg, file, line, func,
        stmt);
    throw std::runtime_error(fmt::format("ERROR[{}] because of {} on {}::{} In {}:({}) FAILED", error_string<T>(err),
                                         msg, file, line, func, stmt));
#endif // defined(__CUDA_ARCH__)
    return true;
  }

  template <>
   inline bool throw_if_error<const char *>(const char *err, const char *msg, const char *stmt, const char *file,
                                                  const char *func, int line) {
#if defined(__CUDA_ARCH__)
    // in device code
    printf("ERROR on %s::%d In %s:(%s) FAILED", file, line, func, stmt);
#else  // defined(__CUDA_ARCH__)
    // in host code
    if (msg == nullptr) {
      LOG(critical, "ERROR[{}] on {}::{} In {}:({}) FAILED", err, file, line, func, stmt);
      throw std::runtime_error(fmt::format("ERROR[{}] on {}::{} In {}:({}) FAILED", err, file, line, func, stmt));
    }
    LOG(critical, "ERROR[{}] because of {} on {}::{} In {}:({}) FAILED", err, msg, file, line, func, stmt);
    throw std::runtime_error(
        fmt::format("ERROR[{}] because of {} on {}::{} In {}:({}) FAILED", err, msg, file, line, func, stmt));
#endif // defined(__CUDA_ARCH__)
    return true;
  }

  template <>
  inline bool throw_if_error<std::string>(std::string err, const char *msg, const char *stmt, const char *file,
                                                 const char *func, int line) {
#if defined(__CUDA_ARCH__)
    // in device code
    printf("ERROR on %s::%d In %s:(%s) FAILED", file, line, func, stmt);
#else  // defined(__CUDA_ARCH__)
    // in host code
    if (msg == nullptr) {
      LOG(critical, "ERROR[{}] on {}::{} In {}:({}) FAILED", err, file, line, func, stmt);
      throw std::runtime_error(fmt::format("ERROR[{}] on {}::{} In {}:({}) FAILED", err, file, line, func, stmt));
    }
    LOG(critical, "ERROR[{}] because of {} on {}::{} In {}:({}) FAILED", err, msg, file, line, func, stmt);
    throw std::runtime_error(
        fmt::format("ERROR[{}] because of {} on {}::{} In {}:({}) FAILED", err, msg, file, line, func, stmt));
#endif // defined(__CUDA_ARCH__)
    return true;
  }



#ifndef IS_ERROR
#define IS_ERROR(stmt) is_error(stmt)
#endif // IS_ERROR

#ifndef PRINT_IF_ERROR
#define PRINT_IF_ERROR(stmt) print_if_error((stmt), #stmt, __FILE__, __func__, __LINE__)
#endif // PRINT_IF_ERROR

#ifndef THROW_IF_ERROR_WITH_MSG
#define THROW_IF_ERROR_WITH_MSG(stmt, msg) throw_if_error(stmt, msg, #stmt, __FILE__, __func__, __LINE__)
#endif // THROW_IF_ERROR_WITH_MSG

#ifndef THROW_IF_ERROR
#define THROW_IF_ERROR(stmt) THROW_IF_ERROR_WITH_MSG(stmt, nullptr)
#endif // PRINT_IF_ERROR
