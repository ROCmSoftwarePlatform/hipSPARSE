/* ************************************************************************
* Copyright (c) 2020 Advanced Micro Devices, Inc.
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
* THE SOFTWARE.
*
* ************************************************************************ */

#pragma once
#ifndef TESTING_PRUNE_DENSE2CSR_BY_PERCENTAGE_HPP
#define TESTING_PRUNE_DENSE2CSR_BY_PERCENTAGE_HPP

#include "hipsparse.hpp"
#include "hipsparse_test_unique_ptr.hpp"
#include "unit.hpp"
#include "utility.hpp"

#include <algorithm>
#include <hipsparse.h>
#include <string>

using namespace hipsparse;
using namespace hipsparse_test;

template <typename T>
void testing_prune_dense2csr_by_percentage_bad_arg(void)
{
    size_t safe_size = 100;

    int    M                      = 10;
    int    N                      = 10;
    int    LDA                    = M;
    T      percentage             = static_cast<T>(1);
    int    nnz_total_dev_host_ptr = 100;
    size_t buffer_size            = 100;

    hipsparseStatus_t status;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    std::unique_ptr<descr_struct> unique_ptr_descr(new descr_struct);
    hipsparseMatDescr_t           descr = unique_ptr_descr->descr;

    std::unique_ptr<prune_struct> unique_ptr_info(new prune_struct);
    pruneInfo_t                   info = unique_ptr_info->info;

    auto csr_row_ptr_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto csr_col_ind_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto csr_val_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
    auto A_managed       = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
    auto temp_buffer_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};

    int* csr_row_ptr = (int*)csr_row_ptr_managed.get();
    int* csr_col_ind = (int*)csr_col_ind_managed.get();
    T*   csr_val     = (T*)csr_val_managed.get();
    T*   A           = (T*)A_managed.get();
    T*   temp_buffer = (T*)temp_buffer_managed.get();

    if(!csr_row_ptr || !csr_col_ind || !csr_val || !A || !temp_buffer)
    {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

#if(!defined(CUDART_VERSION))
    // Test hipsparseXpruneDense2csrByPercentage_bufferSize
    status = hipsparseXpruneDense2csrByPercentage_bufferSize(nullptr,
                                                             M,
                                                             N,
                                                             A,
                                                             LDA,
                                                             percentage,
                                                             descr,
                                                             csr_val,
                                                             csr_row_ptr,
                                                             csr_col_ind,
                                                             info,
                                                             &buffer_size);
    verify_hipsparse_status_invalid_handle(status);

    status = hipsparseXpruneDense2csrByPercentage_bufferSize(
        handle, M, N, A, LDA, percentage, descr, csr_val, csr_row_ptr, csr_col_ind, info, nullptr);
    verify_hipsparse_status_invalid_pointer(status, "Error: buffer size is nullptr");

    // Test hipsparseXpruneDense2csrNnzByPercentage
    status = hipsparseXpruneDense2csrNnzByPercentage(nullptr,
                                                     M,
                                                     N,
                                                     A,
                                                     LDA,
                                                     percentage,
                                                     descr,
                                                     csr_row_ptr,
                                                     &nnz_total_dev_host_ptr,
                                                     info,
                                                     temp_buffer);
    verify_hipsparse_status_invalid_handle(status);

    status = hipsparseXpruneDense2csrNnzByPercentage(handle,
                                                     -1,
                                                     N,
                                                     A,
                                                     LDA,
                                                     percentage,
                                                     descr,
                                                     csr_row_ptr,
                                                     &nnz_total_dev_host_ptr,
                                                     info,
                                                     temp_buffer);
    verify_hipsparse_status_invalid_size(status, "Error: M is invalid");

    status = hipsparseXpruneDense2csrNnzByPercentage(handle,
                                                     M,
                                                     -1,
                                                     A,
                                                     LDA,
                                                     percentage,
                                                     descr,
                                                     csr_row_ptr,
                                                     &nnz_total_dev_host_ptr,
                                                     info,
                                                     temp_buffer);
    verify_hipsparse_status_invalid_size(status, "Error: N is invalid");

    status = hipsparseXpruneDense2csrNnzByPercentage(handle,
                                                     M,
                                                     N,
                                                     A,
                                                     -1,
                                                     percentage,
                                                     descr,
                                                     csr_row_ptr,
                                                     &nnz_total_dev_host_ptr,
                                                     info,
                                                     temp_buffer);
    verify_hipsparse_status_invalid_size(status, "Error: LDA is invalid");

    status = hipsparseXpruneDense2csrNnzByPercentage(handle,
                                                     M,
                                                     N,
                                                     A,
                                                     LDA,
                                                     (T)-1.0,
                                                     descr,
                                                     csr_row_ptr,
                                                     &nnz_total_dev_host_ptr,
                                                     info,
                                                     temp_buffer);
    verify_hipsparse_status_invalid_pointer(status, "Error: percentage is less than 0");

    status = hipsparseXpruneDense2csrNnzByPercentage(handle,
                                                     M,
                                                     N,
                                                     A,
                                                     LDA,
                                                     (T)101.0,
                                                     descr,
                                                     csr_row_ptr,
                                                     &nnz_total_dev_host_ptr,
                                                     info,
                                                     temp_buffer);
    verify_hipsparse_status_invalid_pointer(status, "Error: percentage is more than 100");

    status = hipsparseXpruneDense2csrNnzByPercentage(handle,
                                                     M,
                                                     N,
                                                     (const T*)nullptr,
                                                     LDA,
                                                     percentage,
                                                     descr,
                                                     csr_row_ptr,
                                                     &nnz_total_dev_host_ptr,
                                                     info,
                                                     temp_buffer);
    verify_hipsparse_status_invalid_pointer(status, "Error: A is nullptr");

    status = hipsparseXpruneDense2csrNnzByPercentage(handle,
                                                     M,
                                                     N,
                                                     A,
                                                     LDA,
                                                     percentage,
                                                     nullptr,
                                                     csr_row_ptr,
                                                     &nnz_total_dev_host_ptr,
                                                     info,
                                                     temp_buffer);
    verify_hipsparse_status_invalid_pointer(status, "Error: descr is nullptr");

    status = hipsparseXpruneDense2csrNnzByPercentage(handle,
                                                     M,
                                                     N,
                                                     A,
                                                     LDA,
                                                     percentage,
                                                     descr,
                                                     nullptr,
                                                     &nnz_total_dev_host_ptr,
                                                     info,
                                                     temp_buffer);
    verify_hipsparse_status_invalid_pointer(status, "Error: csr_row_ptr is nullptr");

    status = hipsparseXpruneDense2csrNnzByPercentage(
        handle, M, N, A, LDA, percentage, descr, csr_row_ptr, nullptr, info, temp_buffer);
    verify_hipsparse_status_invalid_pointer(status, "Error: nnz_total_dev_host_ptr is nullptr");

    status = hipsparseXpruneDense2csrNnzByPercentage(handle,
                                                     M,
                                                     N,
                                                     A,
                                                     LDA,
                                                     percentage,
                                                     descr,
                                                     csr_row_ptr,
                                                     &nnz_total_dev_host_ptr,
                                                     info,
                                                     nullptr);
    verify_hipsparse_status_invalid_pointer(status, "Error: buffer size is nullptr");

    // Test hipsparseXpruneDense2csrByPercentage
    status = hipsparseXpruneDense2csrByPercentage(nullptr,
                                                  M,
                                                  N,
                                                  A,
                                                  LDA,
                                                  percentage,
                                                  descr,
                                                  csr_val,
                                                  csr_row_ptr,
                                                  csr_col_ind,
                                                  info,
                                                  temp_buffer);
    verify_hipsparse_status_invalid_handle(status);

    status = hipsparseXpruneDense2csrByPercentage(handle,
                                                  -1,
                                                  N,
                                                  A,
                                                  LDA,
                                                  percentage,
                                                  descr,
                                                  csr_val,
                                                  csr_row_ptr,
                                                  csr_col_ind,
                                                  info,
                                                  temp_buffer);
    verify_hipsparse_status_invalid_size(status, "Error: M is invalid");

    status = hipsparseXpruneDense2csrByPercentage(handle,
                                                  M,
                                                  -1,
                                                  A,
                                                  LDA,
                                                  percentage,
                                                  descr,
                                                  csr_val,
                                                  csr_row_ptr,
                                                  csr_col_ind,
                                                  info,
                                                  temp_buffer);
    verify_hipsparse_status_invalid_size(status, "Error: N is invalid");

    status = hipsparseXpruneDense2csrByPercentage(handle,
                                                  M,
                                                  N,
                                                  A,
                                                  -1,
                                                  percentage,
                                                  descr,
                                                  csr_val,
                                                  csr_row_ptr,
                                                  csr_col_ind,
                                                  info,
                                                  temp_buffer);
    verify_hipsparse_status_invalid_size(status, "Error: LDA is invalid");

    status = hipsparseXpruneDense2csrByPercentage(
        handle, M, N, A, LDA, (T)-1.0, descr, csr_val, csr_row_ptr, csr_col_ind, info, temp_buffer);
    verify_hipsparse_status_invalid_pointer(status, "Error: percentage is less than 0");

    status = hipsparseXpruneDense2csrByPercentage(handle,
                                                  M,
                                                  N,
                                                  A,
                                                  LDA,
                                                  (T)101.0,
                                                  descr,
                                                  csr_val,
                                                  csr_row_ptr,
                                                  csr_col_ind,
                                                  info,
                                                  temp_buffer);
    verify_hipsparse_status_invalid_pointer(status, "Error: percentage is more than 100");

    status = hipsparseXpruneDense2csrByPercentage(handle,
                                                  M,
                                                  N,
                                                  (const T*)nullptr,
                                                  LDA,
                                                  percentage,
                                                  descr,
                                                  csr_val,
                                                  csr_row_ptr,
                                                  csr_col_ind,
                                                  info,
                                                  temp_buffer);
    verify_hipsparse_status_invalid_pointer(status, "Error: A is nullptr");

    status = hipsparseXpruneDense2csrByPercentage(handle,
                                                  M,
                                                  N,
                                                  A,
                                                  LDA,
                                                  percentage,
                                                  nullptr,
                                                  csr_val,
                                                  csr_row_ptr,
                                                  csr_col_ind,
                                                  info,
                                                  temp_buffer);
    verify_hipsparse_status_invalid_pointer(status, "Error: descr is nullptr");

    status = hipsparseXpruneDense2csrByPercentage(handle,
                                                  M,
                                                  N,
                                                  A,
                                                  LDA,
                                                  percentage,
                                                  descr,
                                                  (T*)nullptr,
                                                  csr_row_ptr,
                                                  csr_col_ind,
                                                  info,
                                                  temp_buffer);
    verify_hipsparse_status_invalid_pointer(status, "Error: csr_val is nullptr");

    status = hipsparseXpruneDense2csrByPercentage(
        handle, M, N, A, LDA, percentage, descr, csr_val, nullptr, csr_col_ind, info, temp_buffer);
    verify_hipsparse_status_invalid_pointer(status, "Error: csr_row_ptr is nullptr");

    status = hipsparseXpruneDense2csrByPercentage(
        handle, M, N, A, LDA, percentage, descr, csr_val, csr_row_ptr, nullptr, info, temp_buffer);
    verify_hipsparse_status_invalid_pointer(status, "Error: csr_col_ind is nullptr");

    status = hipsparseXpruneDense2csrByPercentage(
        handle, M, N, A, LDA, percentage, descr, csr_val, csr_row_ptr, csr_col_ind, info, nullptr);
    verify_hipsparse_status_invalid_pointer(status, "Error: buffer is nullptr");
#endif
}

template <typename T>
hipsparseStatus_t testing_prune_dense2csr_by_percentage(Arguments argus)
{
    int                  M          = argus.M;
    int                  N          = argus.N;
    int                  LDA        = argus.lda;
    T                    percentage = static_cast<T>(argus.percentage);
    hipsparseIndexBase_t idx_base   = argus.idx_base;
    hipsparseStatus_t    status;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t              handle = unique_ptr_handle->handle;

    std::unique_ptr<descr_struct> unique_ptr_descr(new descr_struct);
    hipsparseMatDescr_t           descr = unique_ptr_descr->descr;

    std::unique_ptr<prune_struct> unique_ptr_info(new prune_struct);
    pruneInfo_t                   info = unique_ptr_info->info;

    CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));
    CHECK_HIPSPARSE_ERROR(hipsparseSetMatIndexBase(descr, idx_base));

    // Argument sanity check before allocating invalid memory
    if(M <= 0 || N <= 0 || LDA < M || percentage < 0.0 || percentage > 100.0)
    {
	// cusparse does not seem to check LDA < M
#if(defined(CUDART_VERSION))
	if(LDA < M)
	{
	    return HIPSPARSE_STATUS_SUCCESS;
	}
#endif
        size_t safe_size = 100;

        auto csr_row_ptr_managed
            = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
        auto csr_col_ind_managed
            = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
        auto csr_val_managed
            = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
        auto A_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
        auto temp_buffer_managed
            = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};

        int* csr_row_ptr = (int*)csr_row_ptr_managed.get();
        int* csr_col_ind = (int*)csr_col_ind_managed.get();
        T*   csr_val     = (T*)csr_val_managed.get();
        T*   A           = (T*)A_managed.get();
        T*   temp_buffer = (T*)temp_buffer_managed.get();

        if(!csr_row_ptr || !csr_col_ind || !csr_val || !A || !temp_buffer)
        {
            PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
            return HIPSPARSE_STATUS_ALLOC_FAILED;
        }

        status = hipsparseXpruneDense2csrByPercentage(handle,
                                                      M,
                                                      N,
                                                      A,
                                                      LDA,
                                                      percentage,
                                                      descr,
                                                      csr_val,
                                                      csr_row_ptr,
                                                      csr_col_ind,
                                                      info,
                                                      temp_buffer);

        if(M < 0 || N < 0 || LDA < M || percentage < 0.0 || percentage > 100.0)
        {
            verify_hipsparse_status_invalid_size(
                status,
                "Error: m < 0 || n < 0 || lda < m || percentage < 0.0 || percentage > 100.0");
        }
        else
        {
            verify_hipsparse_status_success(
                status, "m >= 0 && n >= 0 && lda >= m && percentage >= 0.0 && percentage <= 100.0");
        }

        return HIPSPARSE_STATUS_SUCCESS;
    }

    // Allocate host memory
    std::vector<T>   h_A(LDA * N);
    std::vector<int> h_nnz_total_dev_host_ptr(1);

    // Allocate device memory
    auto d_A_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * LDA * N), device_free};
    auto d_nnz_total_dev_host_ptr_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int)), device_free};
    auto d_csr_row_ptr_managed
        = hipsparse_unique_ptr{device_malloc(sizeof(int) * (M + 1)), device_free};

    T*   d_A                      = (T*)d_A_managed.get();
    int* d_nnz_total_dev_host_ptr = (int*)d_nnz_total_dev_host_ptr_managed.get();
    int* d_csr_row_ptr            = (int*)d_csr_row_ptr_managed.get();

    if(!d_A || !d_nnz_total_dev_host_ptr || !d_csr_row_ptr)
    {
        verify_hipsparse_status_success(HIPSPARSE_STATUS_ALLOC_FAILED,
                                        "!d_A || !d_nnz_total_dev_host_ptr || !d_csr_row_ptr");
        return HIPSPARSE_STATUS_ALLOC_FAILED;
    }

    // Initialize the entire allocated memory.
    for(int i = 0; i < LDA; ++i)
    {
        for(int j = 0; j < N; ++j)
        {
            h_A[j * LDA + i] = make_DataType<T>(-1);
        }
    }

    // Initialize a random dense matrix.
    srand(0);
    gen_dense_random_sparsity_pattern(M, N, h_A.data(), LDA, 0.2);

    // Transfer.
    CHECK_HIP_ERROR(hipMemcpy(d_A, h_A.data(), sizeof(T) * LDA * N, hipMemcpyHostToDevice));

    size_t buffer_size = 512;
    CHECK_HIPSPARSE_ERROR(hipsparseXpruneDense2csrByPercentage_bufferSize(handle,
                                                                          M,
                                                                          N,
                                                                          d_A,
                                                                          LDA,
                                                                          percentage,
                                                                          descr,
                                                                          (const T*)nullptr,
                                                                          d_csr_row_ptr,
                                                                          (const int*)nullptr,
                                                                          info,
                                                                          &buffer_size));

    auto d_temp_buffer_managed = hipsparse_unique_ptr{device_malloc(buffer_size), device_free};

    T* d_temp_buffer = (T*)d_temp_buffer_managed.get();

    if(!d_temp_buffer)
    {
        verify_hipsparse_status_success(HIPSPARSE_STATUS_ALLOC_FAILED, "!d_temp_buffer");
        return HIPSPARSE_STATUS_ALLOC_FAILED;
    }

    CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));
    CHECK_HIPSPARSE_ERROR(hipsparseXpruneDense2csrNnzByPercentage(handle,
                                                                  M,
                                                                  N,
                                                                  d_A,
                                                                  LDA,
                                                                  percentage,
                                                                  descr,
                                                                  d_csr_row_ptr,
                                                                  &h_nnz_total_dev_host_ptr[0],
                                                                  info,
                                                                  d_temp_buffer));

    CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_DEVICE));
    CHECK_HIPSPARSE_ERROR(hipsparseXpruneDense2csrNnzByPercentage(handle,
                                                                  M,
                                                                  N,
                                                                  d_A,
                                                                  LDA,
                                                                  percentage,
                                                                  descr,
                                                                  d_csr_row_ptr,
                                                                  d_nnz_total_dev_host_ptr,
                                                                  info,
                                                                  d_temp_buffer));

    if(argus.unit_check)
    {
        std::vector<int> h_nnz_total_copied_from_device(1);
        CHECK_HIP_ERROR(hipMemcpy(h_nnz_total_copied_from_device.data(),
                                  d_nnz_total_dev_host_ptr,
                                  sizeof(int),
                                  hipMemcpyDeviceToHost));

        unit_check_general<int>(
            1, 1, 1, h_nnz_total_dev_host_ptr.data(), h_nnz_total_copied_from_device.data());

        CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));

        if(h_nnz_total_dev_host_ptr[0] > 0)
        {
            auto d_csr_col_ind_managed = hipsparse_unique_ptr{
                device_malloc(sizeof(int) * h_nnz_total_dev_host_ptr[0]), device_free};
            auto d_csr_val_managed = hipsparse_unique_ptr{
                device_malloc(sizeof(T) * h_nnz_total_dev_host_ptr[0]), device_free};

            int* d_csr_col_ind = (int*)d_csr_col_ind_managed.get();
            T*   d_csr_val     = (T*)d_csr_val_managed.get();

            if(!d_csr_col_ind || !d_csr_val)
            {
                verify_hipsparse_status_success(HIPSPARSE_STATUS_ALLOC_FAILED,
                                                "!d_csr_col_ind || !d_csr_val");
                return HIPSPARSE_STATUS_ALLOC_FAILED;
            }

            CHECK_HIPSPARSE_ERROR(hipsparseXpruneDense2csrByPercentage(handle,
                                                                       M,
                                                                       N,
                                                                       d_A,
                                                                       LDA,
                                                                       percentage,
                                                                       descr,
                                                                       d_csr_val,
                                                                       d_csr_row_ptr,
                                                                       d_csr_col_ind,
                                                                       info,
                                                                       d_temp_buffer));

            std::vector<int> h_csr_row_ptr(M + 1);
            std::vector<int> h_csr_col_ind(h_nnz_total_dev_host_ptr[0]);
            std::vector<T>   h_csr_val(h_nnz_total_dev_host_ptr[0]);

            CHECK_HIP_ERROR(hipMemcpy(
                h_csr_row_ptr.data(), d_csr_row_ptr, sizeof(int) * (M + 1), hipMemcpyDeviceToHost));

            CHECK_HIP_ERROR(hipMemcpy(h_csr_col_ind.data(),
                                      d_csr_col_ind,
                                      sizeof(int) * h_nnz_total_dev_host_ptr[0],
                                      hipMemcpyDeviceToHost));
            CHECK_HIP_ERROR(hipMemcpy(h_csr_val.data(),
                                      d_csr_val,
                                      sizeof(T) * h_nnz_total_dev_host_ptr[0],
                                      hipMemcpyDeviceToHost));

            // call host and check results
            std::vector<int> h_nnz_cpu(1);
            std::vector<int> h_csr_row_ptr_cpu;
            std::vector<int> h_csr_col_ind_cpu;
            std::vector<T>   h_csr_val_cpu;

            host_prune_dense2csr_by_percentage(M,
                                               N,
                                               h_A,
                                               LDA,
                                               idx_base,
                                               percentage,
                                               h_nnz_cpu[0],
                                               h_csr_val_cpu,
                                               h_csr_row_ptr_cpu,
                                               h_csr_col_ind_cpu);

            unit_check_general<int>(1, 1, 1, h_nnz_cpu.data(), h_nnz_total_dev_host_ptr.data());
            unit_check_general<int>(1, (M + 1), 1, h_csr_row_ptr_cpu.data(), h_csr_row_ptr.data());
            unit_check_general<int>(
                1, h_nnz_total_dev_host_ptr[0], 1, h_csr_col_ind_cpu.data(), h_csr_col_ind.data());
            unit_check_general<T>(
                1, h_nnz_total_dev_host_ptr[0], 1, h_csr_val_cpu.data(), h_csr_val.data());
        }
    }

    return HIPSPARSE_STATUS_SUCCESS;
}

#endif // TESTING_PRUNE_DENSE2CSR_BY_PERCENTAGE_HPP
