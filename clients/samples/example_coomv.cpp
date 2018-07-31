/* ************************************************************************
 * Copyright 2018 Advanced Micro Devices, Inc.
 * ************************************************************************ */

#include "utility.hpp"

#include <stdlib.h>
#include <stdio.h>
#include <vector>
#include <hipsparse.h>
#include <hip/hip_runtime_api.h>

int main(int argc, char* argv[])
{
    // Parse command line
    if(argc < 2)
    {
        fprintf(stderr, "%s <ndim> [<trials> <batch_size>]\n", argv[0]);
        return -1;
    }

    int ndim       = atoi(argv[1]);
    int trials     = 200;
    int batch_size = 1;

    if(argc > 2)
    {
        trials = atoi(argv[2]);
    }
    if(argc > 3)
    {
        batch_size = atoi(argv[3]);
    }

    // hipSPARSE handle
    hipsparseHandle_t handle;
    hipsparseCreate(&handle);

    hipDeviceProp_t devProp;
    int device_id = 0;

    hipGetDevice(&device_id);
    hipGetDeviceProperties(&devProp, device_id);
    printf("Device: %s\n", devProp.name);

    // Generate problem
    std::vector<int> hAptr;
    std::vector<int> hAcol;
    std::vector<double> hAval;
    int m   = gen_2d_laplacian(ndim, hAptr, hAcol, hAval, HIPSPARSE_INDEX_BASE_ZERO);
    int n   = m;
    int nnz = hAptr[m];

    // Convert to COO matrix
    std::vector<int> hArow(nnz);

    for(int i = 0; i < m; ++i)
    {
        for(int j = hAptr[i]; j < hAptr[i + 1]; ++j)
        {
            hArow[j] = i;
        }
    }

    // Sample some random data
    srand(12345ULL);

    double halpha = static_cast<double>(rand()) / RAND_MAX;
    double hbeta  = 0.0;

    std::vector<double> hx(n);
    hipsparseInit(hx, 1, n);

    // Matrix descriptor
    hipsparseMatDescr_t descrA;
    hipsparseCreateMatDescr(&descrA);

    // Offload data to device
    int* dArow    = NULL;
    int* dAcol    = NULL;
    double* dAval = NULL;
    double* dx    = NULL;
    double* dy    = NULL;

    hipMalloc((void**)&dArow, sizeof(int) * nnz);
    hipMalloc((void**)&dAcol, sizeof(int) * nnz);
    hipMalloc((void**)&dAval, sizeof(double) * nnz);
    hipMalloc((void**)&dx, sizeof(double) * n);
    hipMalloc((void**)&dy, sizeof(double) * m);

    hipMemcpy(dArow, hArow.data(), sizeof(int) * nnz, hipMemcpyHostToDevice);
    hipMemcpy(dAcol, hAcol.data(), sizeof(int) * nnz, hipMemcpyHostToDevice);
    hipMemcpy(dAval, hAval.data(), sizeof(double) * nnz, hipMemcpyHostToDevice);
    hipMemcpy(dx, hx.data(), sizeof(double) * n, hipMemcpyHostToDevice);

    // Warm up
    for(int i = 0; i < 10; ++i)
    {
        // Call hipsparse coomv
        hipsparseDcoomv(handle,
                        HIPSPARSE_OPERATION_NON_TRANSPOSE,
                        m,
                        n,
                        nnz,
                        &halpha,
                        descrA,
                        dAval,
                        dArow,
                        dAcol,
                        dx,
                        &hbeta,
                        dy);
    }

    // Device synchronization
    hipDeviceSynchronize();

    // Start time measurement
    double time = get_time_us();

    // COO matrix vector multiplication
    for(int i = 0; i < trials; ++i)
    {
        for(int i = 0; i < batch_size; ++i)
        {
            // Call hipsparse coomv
            hipsparseDcoomv(handle,
                            HIPSPARSE_OPERATION_NON_TRANSPOSE,
                            m,
                            n,
                            nnz,
                            &halpha,
                            descrA,
                            dAval,
                            dArow,
                            dAcol,
                            dx,
                            &hbeta,
                            dy);
        }

        // Device synchronization
        hipDeviceSynchronize();
    }

    time = (get_time_us() - time) / (trials * batch_size * 1e3);
    double bandwidth =
        static_cast<double>(sizeof(double) * (4 * m + nnz) + sizeof(int) * (2 * nnz)) / time / 1e6;
    double gflops = static_cast<double>(3 * nnz) / time / 1e6;
    printf("m\t\tn\t\tnnz\t\talpha\tbeta\tGFlops\tGB/s\tusec\n");
    printf("%8d\t%8d\t%9d\t%0.2lf\t%0.2lf\t%0.2lf\t%0.2lf\t%0.2lf\n",
           m,
           n,
           nnz,
           halpha,
           hbeta,
           gflops,
           bandwidth,
           time);

    // Clear up on device
    hipFree(dArow);
    hipFree(dAcol);
    hipFree(dAval);
    hipFree(dx);
    hipFree(dy);

    hipsparseDestroyMatDescr(descrA);
    hipsparseDestroy(handle);

    return 0;
}
