/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#pragma once
#ifndef TESTING_ROTI_HPP
#define TESTING_ROTI_HPP

#include "hipsparse_test_unique_ptr.hpp"
#include "hipsparse.hpp"
#include "utility.hpp"
#include "unit.hpp"

#include <hipsparse.h>

using namespace hipsparse;
using namespace hipsparse_test;

template <typename T>
void testing_roti_bad_arg(void)
{
    int nnz       = 100;
    int safe_size = 100;
    T c           = 3.7;
    T s           = 1.2;

    hipsparseIndexBase_t idx_base = HIPSPARSE_INDEX_BASE_ZERO;
    hipsparseStatus_t status;

    std::unique_ptr<handle_struct> unique_ptr_handle(new handle_struct);
    hipsparseHandle_t handle = unique_ptr_handle->handle;

    auto dx_val_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
    auto dx_ind_managed = hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
    auto dy_managed     = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};

    T* dx_val   = (T*)dx_val_managed.get();
    int* dx_ind = (int*)dx_ind_managed.get();
    T* dy       = (T*)dy_managed.get();

    if(!dx_ind || !dx_val || !dy)
    {
        PRINT_IF_HIP_ERROR(hipErrorOutOfMemory);
        return;
    }

    // testing for(nullptr == dx_ind)
    {
        int* dx_ind_null = nullptr;

        status = hipsparseXroti(handle, nnz, dx_val, dx_ind_null, dy, &c, &s, idx_base);
        verify_hipsparse_status_invalid_pointer(status, "Error: x_ind is nullptr");
    }
    // testing for(nullptr == dx_val)
    {
        T* dx_val_null = nullptr;

        status = hipsparseXroti(handle, nnz, dx_val_null, dx_ind, dy, &c, &s, idx_base);
        verify_hipsparse_status_invalid_pointer(status, "Error: x_val is nullptr");
    }
    // testing for(nullptr == dy)
    {
        T* dy_null = nullptr;

        status = hipsparseXroti(handle, nnz, dx_val, dx_ind, dy_null, &c, &s, idx_base);
        verify_hipsparse_status_invalid_pointer(status, "Error: y is nullptr");
    }
    // testing for(nullptr == c)
    {
        T* dc_null = nullptr;

        status = hipsparseXroti(handle, nnz, dx_val, dx_ind, dy, dc_null, &s, idx_base);
        verify_hipsparse_status_invalid_pointer(status, "Error: c is nullptr");
    }
    // testing for(nullptr == s)
    {
        T* ds_null = nullptr;

        status = hipsparseXroti(handle, nnz, dx_val, dx_ind, dy, &c, ds_null, idx_base);
        verify_hipsparse_status_invalid_pointer(status, "Error: s is nullptr");
    }
    // testing for(nullptr == handle)
    {
        hipsparseHandle_t handle_null = nullptr;

        status = hipsparseXroti(handle_null, nnz, dx_val, dx_ind, dy, &c, &s, idx_base);
        verify_hipsparse_status_invalid_handle(status);
    }
}

template <typename T>
hipsparseStatus_t testing_roti(Arguments argus)
{
    int N                         = argus.N;
    int nnz                       = argus.nnz;
    T c                           = argus.alpha;
    T s                           = argus.beta;
    int safe_size                 = 100;
    hipsparseIndexBase_t idx_base = argus.idx_base;
    hipsparseStatus_t status;

    std::unique_ptr<handle_struct> test_handle(new handle_struct);
    hipsparseHandle_t handle = test_handle->handle;

    // Argument sanity check before allocating invalid memory
    if(nnz <= 0)
    {
        auto dx_ind_managed =
            hipsparse_unique_ptr{device_malloc(sizeof(int) * safe_size), device_free};
        auto dx_val_managed =
            hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};
        auto dy_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * safe_size), device_free};

        int* dx_ind = (int*)dx_ind_managed.get();
        T* dx_val   = (T*)dx_val_managed.get();
        T* dy       = (T*)dy_managed.get();

        if(!dx_ind || !dx_val || !dy)
        {
            verify_hipsparse_status_success(HIPSPARSE_STATUS_ALLOC_FAILED,
                                            "!dx_ind || !dx_val || !dy");
            return HIPSPARSE_STATUS_ALLOC_FAILED;
        }

        CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));
        status = hipsparseXroti(handle, nnz, dx_val, dx_ind, dy, &c, &s, idx_base);

        if(nnz < 0)
        {
            verify_hipsparse_status_invalid_size(status, "Error: nnz < 0");
        }
        else
        {
            verify_hipsparse_status_success(status, "nnz == 0");
        }

        return HIPSPARSE_STATUS_SUCCESS;
    }

    // Host structures
    std::vector<int> hx_ind(nnz);
    std::vector<T> hx_val_1(nnz);
    std::vector<T> hx_val_2(nnz);
    std::vector<T> hx_val_gold(nnz);
    std::vector<T> hy_1(N);
    std::vector<T> hy_2(N);
    std::vector<T> hy_gold(N);

    // Initial Data on CPU
    srand(12345ULL);
    hipsparseInitIndex(hx_ind.data(), nnz, 1, N);
    hipsparseInit<T>(hx_val_1, 1, nnz);
    hipsparseInit<T>(hy_1, 1, N);

    hx_val_2    = hx_val_1;
    hx_val_gold = hx_val_1;
    hy_2        = hy_1;
    hy_gold     = hy_1;

    // allocate memory on device
    auto dx_ind_managed   = hipsparse_unique_ptr{device_malloc(sizeof(int) * nnz), device_free};
    auto dx_val_1_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz), device_free};
    auto dx_val_2_managed = hipsparse_unique_ptr{device_malloc(sizeof(T) * nnz), device_free};
    auto dy_1_managed     = hipsparse_unique_ptr{device_malloc(sizeof(T) * N), device_free};
    auto dy_2_managed     = hipsparse_unique_ptr{device_malloc(sizeof(T) * N), device_free};
    auto dc_managed       = hipsparse_unique_ptr{device_malloc(sizeof(T)), device_free};
    auto ds_managed       = hipsparse_unique_ptr{device_malloc(sizeof(T)), device_free};

    int* dx_ind = (int*)dx_ind_managed.get();
    T* dx_val_1 = (T*)dx_val_1_managed.get();
    T* dx_val_2 = (T*)dx_val_2_managed.get();
    T* dy_1     = (T*)dy_1_managed.get();
    T* dy_2     = (T*)dy_2_managed.get();
    T* dc       = (T*)dc_managed.get();
    T* ds       = (T*)ds_managed.get();

    if(!dx_ind || !dx_val_1 || !dx_val_2 || !dy_1 || !dy_2 || !dc || !ds)
    {
        verify_hipsparse_status_success(
            HIPSPARSE_STATUS_ALLOC_FAILED,
            "!dx_ind || !dx_val_1 || !dx_val_2 || !dy_1 || !dy_2 || !dc || !ds");
        return HIPSPARSE_STATUS_ALLOC_FAILED;
    }

    // copy data from CPU to device
    CHECK_HIP_ERROR(hipMemcpy(dx_ind, hx_ind.data(), sizeof(int) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dx_val_1, hx_val_1.data(), sizeof(T) * nnz, hipMemcpyHostToDevice));
    CHECK_HIP_ERROR(hipMemcpy(dy_1, hy_1.data(), sizeof(T) * N, hipMemcpyHostToDevice));

    if(argus.unit_check)
    {
        CHECK_HIP_ERROR(
            hipMemcpy(dx_val_2, hx_val_2.data(), sizeof(T) * nnz, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(dy_2, hy_2.data(), sizeof(T) * N, hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(dc, &c, sizeof(T), hipMemcpyHostToDevice));
        CHECK_HIP_ERROR(hipMemcpy(ds, &s, sizeof(T), hipMemcpyHostToDevice));

        // ROCSPARSE pointer mode host
        CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));
        CHECK_HIPSPARSE_ERROR(
            hipsparseXroti(handle, nnz, dx_val_1, dx_ind, dy_1, &c, &s, idx_base));

        // ROCSPARSE pointer mode device
        CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_DEVICE));
        CHECK_HIPSPARSE_ERROR(
            hipsparseXroti(handle, nnz, dx_val_2, dx_ind, dy_2, dc, ds, idx_base));

        // copy output from device to CPU
        CHECK_HIP_ERROR(
            hipMemcpy(hx_val_1.data(), dx_val_1, sizeof(T) * nnz, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(
            hipMemcpy(hx_val_2.data(), dx_val_2, sizeof(T) * nnz, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hy_1.data(), dy_1, sizeof(T) * N, hipMemcpyDeviceToHost));
        CHECK_HIP_ERROR(hipMemcpy(hy_2.data(), dy_2, sizeof(T) * N, hipMemcpyDeviceToHost));

        // CPU
        double cpu_time_used = get_time_us();

        for(int i = 0; i < nnz; ++i)
        {
            int idx = hx_ind[i] - idx_base;

            T x = hx_val_gold[i];
            T y = hy_gold[idx];

            hx_val_gold[i] = c * x + s * y;
            hy_gold[idx]   = c * y - s * x;
        }

        cpu_time_used = get_time_us() - cpu_time_used;

        // enable unit check, notice unit check is not invasive, but norm check is,
        // unit check and norm check can not be interchanged their order
        unit_check_general(1, nnz, 1, hx_val_gold.data(), hx_val_1.data());
        unit_check_general(1, nnz, 1, hx_val_gold.data(), hx_val_2.data());
        unit_check_general(1, N, 1, hy_gold.data(), hy_1.data());
        unit_check_general(1, N, 1, hy_gold.data(), hy_2.data());
    }

    if(argus.timing)
    {
        int number_cold_calls = 2;
        int number_hot_calls  = argus.iters;
        CHECK_HIPSPARSE_ERROR(hipsparseSetPointerMode(handle, HIPSPARSE_POINTER_MODE_HOST));

        for(int iter = 0; iter < number_cold_calls; iter++)
        {
            hipsparseXroti(handle, nnz, dx_val_1, dx_ind, dy_1, &c, &s, idx_base);
        }

        double gpu_time_used = get_time_us(); // in microseconds

        for(int iter = 0; iter < number_hot_calls; iter++)
        {
            hipsparseXroti(handle, nnz, dx_val_1, dx_ind, dy_1, &c, &s, idx_base);
        }

        gpu_time_used    = (get_time_us() - gpu_time_used) / number_hot_calls;
        double gflops    = nnz * 6.0 / gpu_time_used / 1e3;
        double bandwidth = (sizeof(int) * nnz + sizeof(T) * 2.0 * nnz) / gpu_time_used / 1e3;

        printf("nnz\t\tcosine\tsine\tGFlop/s\tGB/s\tusec\n");
        printf("%9d\t%0.2lf\t%0.2lf\t%0.2lf\t%0.2lf\t%0.2lf\n",
               nnz,
               c,
               s,
               gflops,
               bandwidth,
               gpu_time_used);
    }
    return HIPSPARSE_STATUS_SUCCESS;
}

#endif // TESTING_ROTI_HPP
