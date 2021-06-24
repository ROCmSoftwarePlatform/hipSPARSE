/* ************************************************************************
* Copyright (c) 2018-2021 Advanced Micro Devices, Inc.
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

#include "hipsparse.h"

#include <hip/hip_complex.h>
#include <hip/hip_runtime_api.h>
#include <rocsparse.h>
#include <stdio.h>
#include <stdlib.h>

#include <iostream>

#define TO_STR2(x) #x
#define TO_STR(x) TO_STR2(x)

#ifdef __cplusplus
extern "C" {
#endif

#define RETURN_IF_HIP_ERROR(INPUT_STATUS_FOR_CHECK)                 \
    {                                                               \
        hipError_t TMP_STATUS_FOR_CHECK = INPUT_STATUS_FOR_CHECK;   \
        if(TMP_STATUS_FOR_CHECK != hipSuccess)                      \
        {                                                           \
            return hipErrorToHIPSPARSEStatus(TMP_STATUS_FOR_CHECK); \
        }                                                           \
    }

#define RETURN_IF_HIPSPARSE_ERROR(INPUT_STATUS_FOR_CHECK)                \
    {                                                                    \
        hipsparseStatus_t TMP_STATUS_FOR_CHECK = INPUT_STATUS_FOR_CHECK; \
        if(TMP_STATUS_FOR_CHECK != HIPSPARSE_STATUS_SUCCESS)             \
        {                                                                \
            return TMP_STATUS_FOR_CHECK;                                 \
        }                                                                \
    }

#define RETURN_IF_ROCSPARSE_ERROR(INPUT_STATUS_FOR_CHECK)               \
    {                                                                   \
        rocsparse_status TMP_STATUS_FOR_CHECK = INPUT_STATUS_FOR_CHECK; \
        if(TMP_STATUS_FOR_CHECK != rocsparse_status_success)            \
        {                                                               \
            return rocSPARSEStatusToHIPStatus(TMP_STATUS_FOR_CHECK);    \
        }                                                               \
    }

// csru2csr struct - to hold permutation array
struct csru2csrInfo
{
    int  size = 0;
    int* P    = nullptr;
};

// Functions needed for hipsparse to match cuda API but which not part of rocsparse backend API
extern rocsparse_status rocsparse_dsbsrilu0_numeric_boost(rocsparse_handle   handle,
                                                          rocsparse_mat_info info,
                                                          int                enable_boost,
                                                          const double*      boost_tol,
                                                          const float*       boost_val);

extern rocsparse_status rocsparse_dcbsrilu0_numeric_boost(rocsparse_handle   handle,
                                                          rocsparse_mat_info info,
                                                          int                enable_boost,
                                                          const double*      boost_tol,
                                                          const rocsparse_float_complex* boost_val);

extern rocsparse_status rocsparse_dscsrilu0_numeric_boost(rocsparse_handle   handle,
                                                          rocsparse_mat_info info,
                                                          int                enable_boost,
                                                          const double*      boost_tol,
                                                          const float*       boost_val);

extern rocsparse_status rocsparse_dccsrilu0_numeric_boost(rocsparse_handle   handle,
                                                          rocsparse_mat_info info,
                                                          int                enable_boost,
                                                          const double*      boost_tol,
                                                          const rocsparse_float_complex* boost_val);

extern rocsparse_status rocsparse_isctr(rocsparse_handle     handle,
                                        rocsparse_int        nnz,
                                        const rocsparse_int* x_val,
                                        const rocsparse_int* x_ind,
                                        rocsparse_int*       y,
                                        rocsparse_index_base idx_base);

hipsparseStatus_t hipErrorToHIPSPARSEStatus(hipError_t status)
{
    switch(status)
    {
    case hipSuccess:
        return HIPSPARSE_STATUS_SUCCESS;
    case hipErrorMemoryAllocation:
    case hipErrorLaunchOutOfResources:
        return HIPSPARSE_STATUS_ALLOC_FAILED;
    case hipErrorInvalidDevicePointer:
        return HIPSPARSE_STATUS_INVALID_VALUE;
    case hipErrorInvalidDevice:
    case hipErrorInvalidResourceHandle:
        return HIPSPARSE_STATUS_NOT_INITIALIZED;
    case hipErrorInvalidValue:
        return HIPSPARSE_STATUS_INVALID_VALUE;
    case hipErrorNoDevice:
    case hipErrorUnknown:
        return HIPSPARSE_STATUS_INTERNAL_ERROR;
    default:
        return HIPSPARSE_STATUS_INTERNAL_ERROR;
    }
}

hipsparseStatus_t rocSPARSEStatusToHIPStatus(rocsparse_status_ status)
{
    switch(status)
    {
    case rocsparse_status_success:
        return HIPSPARSE_STATUS_SUCCESS;
    case rocsparse_status_invalid_handle:
        return HIPSPARSE_STATUS_INVALID_VALUE;
    case rocsparse_status_not_implemented:
        return HIPSPARSE_STATUS_INTERNAL_ERROR;
    case rocsparse_status_invalid_pointer:
        return HIPSPARSE_STATUS_INVALID_VALUE;
    case rocsparse_status_invalid_size:
        return HIPSPARSE_STATUS_INVALID_VALUE;
    case rocsparse_status_memory_error:
        return HIPSPARSE_STATUS_ALLOC_FAILED;
    case rocsparse_status_internal_error:
        return HIPSPARSE_STATUS_INTERNAL_ERROR;
    case rocsparse_status_invalid_value:
        return HIPSPARSE_STATUS_INVALID_VALUE;
    case rocsparse_status_arch_mismatch:
        return HIPSPARSE_STATUS_ARCH_MISMATCH;
    case rocsparse_status_zero_pivot:
        return HIPSPARSE_STATUS_ZERO_PIVOT;
    default:
        throw "Non existent rocsparse_status";
    }
}

rocsparse_pointer_mode_ hipPtrModeToHCCPtrMode(hipsparsePointerMode_t mode)
{
    switch(mode)
    {
    case HIPSPARSE_POINTER_MODE_HOST:
        return rocsparse_pointer_mode_host;
    case HIPSPARSE_POINTER_MODE_DEVICE:
        return rocsparse_pointer_mode_device;
    default:
        throw "Non existent hipsparsePointerMode_t";
    }
}

hipsparsePointerMode_t HCCPtrModeToHIPPtrMode(rocsparse_pointer_mode_ mode)
{
    switch(mode)
    {
    case rocsparse_pointer_mode_host:
        return HIPSPARSE_POINTER_MODE_HOST;
    case rocsparse_pointer_mode_device:
        return HIPSPARSE_POINTER_MODE_DEVICE;
    default:
        throw "Non existent rocsparse_pointer_mode";
    }
}

rocsparse_action_ hipActionToHCCAction(hipsparseAction_t action)
{
    switch(action)
    {
    case HIPSPARSE_ACTION_SYMBOLIC:
        return rocsparse_action_symbolic;
    case HIPSPARSE_ACTION_NUMERIC:
        return rocsparse_action_numeric;
    default:
        throw "Non existent hipsparseAction_t";
    }
}

rocsparse_matrix_type_ hipMatTypeToHCCMatType(hipsparseMatrixType_t type)
{
    switch(type)
    {
    case HIPSPARSE_MATRIX_TYPE_GENERAL:
        return rocsparse_matrix_type_general;
    case HIPSPARSE_MATRIX_TYPE_SYMMETRIC:
        return rocsparse_matrix_type_symmetric;
    case HIPSPARSE_MATRIX_TYPE_HERMITIAN:
        return rocsparse_matrix_type_hermitian;
    case HIPSPARSE_MATRIX_TYPE_TRIANGULAR:
        return rocsparse_matrix_type_triangular;
    default:
        throw "Non existent hipsparseMatrixType_t";
    }
}

hipsparseMatrixType_t HCCMatTypeToHIPMatType(rocsparse_matrix_type_ type)
{
    switch(type)
    {
    case rocsparse_matrix_type_general:
        return HIPSPARSE_MATRIX_TYPE_GENERAL;
    case rocsparse_matrix_type_symmetric:
        return HIPSPARSE_MATRIX_TYPE_SYMMETRIC;
    case rocsparse_matrix_type_hermitian:
        return HIPSPARSE_MATRIX_TYPE_HERMITIAN;
    case rocsparse_matrix_type_triangular:
        return HIPSPARSE_MATRIX_TYPE_TRIANGULAR;
    default:
        throw "Non existent rocsparse_matrix_type";
    }
}

rocsparse_fill_mode_ hipFillModeToHCCFillMode(hipsparseFillMode_t fillMode)
{
    switch(fillMode)
    {
    case HIPSPARSE_FILL_MODE_LOWER:
        return rocsparse_fill_mode_lower;
    case HIPSPARSE_FILL_MODE_UPPER:
        return rocsparse_fill_mode_upper;
    default:
        throw "Non existent hipsparseFillMode_t";
    }
}

hipsparseFillMode_t HCCFillModeToHIPFillMode(rocsparse_fill_mode_ fillMode)
{
    switch(fillMode)
    {
    case rocsparse_fill_mode_lower:
        return HIPSPARSE_FILL_MODE_LOWER;
    case rocsparse_fill_mode_upper:
        return HIPSPARSE_FILL_MODE_UPPER;
    default:
        throw "Non existent rocsparse_fill_mode";
    }
}

rocsparse_diag_type_ hipDiagTypeToHCCDiagType(hipsparseDiagType_t diagType)
{
    switch(diagType)
    {
    case HIPSPARSE_DIAG_TYPE_UNIT:
        return rocsparse_diag_type_unit;
    case HIPSPARSE_DIAG_TYPE_NON_UNIT:
        return rocsparse_diag_type_non_unit;
    default:
        throw "Non existent hipsparseDiagType_t";
    }
}

hipsparseDiagType_t HCCDiagTypeToHIPDiagType(rocsparse_diag_type_ diagType)
{
    switch(diagType)
    {
    case rocsparse_diag_type_unit:
        return HIPSPARSE_DIAG_TYPE_UNIT;
    case rocsparse_diag_type_non_unit:
        return HIPSPARSE_DIAG_TYPE_NON_UNIT;
    default:
        throw "Non existent rocsparse_diag_type";
    }
}

rocsparse_index_base_ hipBaseToHCCBase(hipsparseIndexBase_t base)
{
    switch(base)
    {
    case HIPSPARSE_INDEX_BASE_ZERO:
        return rocsparse_index_base_zero;
    case HIPSPARSE_INDEX_BASE_ONE:
        return rocsparse_index_base_one;
    default:
        throw "Non existent hipsparseIndexBase_t";
    }
}

hipsparseIndexBase_t HCCBaseToHIPBase(rocsparse_index_base_ base)
{
    switch(base)
    {
    case rocsparse_index_base_zero:
        return HIPSPARSE_INDEX_BASE_ZERO;
    case rocsparse_index_base_one:
        return HIPSPARSE_INDEX_BASE_ONE;
    default:
        throw "Non existent rocsparse_index_base";
    }
}

rocsparse_operation_ hipOperationToHCCOperation(hipsparseOperation_t op)
{
    switch(op)
    {
    case HIPSPARSE_OPERATION_NON_TRANSPOSE:
        return rocsparse_operation_none;
    case HIPSPARSE_OPERATION_TRANSPOSE:
        return rocsparse_operation_transpose;
    case HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE:
        return rocsparse_operation_conjugate_transpose;
    default:
        throw "Non existent hipsparseOperation_t";
    }
}

hipsparseOperation_t HCCOperationToHIPOperation(rocsparse_operation_ op)
{
    switch(op)
    {
    case rocsparse_operation_none:
        return HIPSPARSE_OPERATION_NON_TRANSPOSE;
    case rocsparse_operation_transpose:
        return HIPSPARSE_OPERATION_TRANSPOSE;
    case rocsparse_operation_conjugate_transpose:
        return HIPSPARSE_OPERATION_CONJUGATE_TRANSPOSE;
    default:
        throw "Non existent rocsparse_operation";
    }
}

rocsparse_hyb_partition_ hipHybPartToHCCHybPart(hipsparseHybPartition_t partition)
{
    switch(partition)
    {
    case HIPSPARSE_HYB_PARTITION_AUTO:
        return rocsparse_hyb_partition_auto;
    case HIPSPARSE_HYB_PARTITION_USER:
        return rocsparse_hyb_partition_user;
    case HIPSPARSE_HYB_PARTITION_MAX:
        return rocsparse_hyb_partition_max;
    default:
        throw "Non existent hipsparseHybPartition_t";
    }
}

hipsparseHybPartition_t HCCHybPartToHIPHybPart(rocsparse_hyb_partition_ partition)
{
    switch(partition)
    {
    case rocsparse_hyb_partition_auto:
        return HIPSPARSE_HYB_PARTITION_AUTO;
    case rocsparse_hyb_partition_user:
        return HIPSPARSE_HYB_PARTITION_USER;
    case rocsparse_hyb_partition_max:
        return HIPSPARSE_HYB_PARTITION_MAX;
    default:
        throw "Non existent rocsparse_hyb_partition";
    }
}

rocsparse_direction_ hipDirectionToHCCDirection(hipsparseDirection_t op)
{
    switch(op)
    {
    case HIPSPARSE_DIRECTION_ROW:
        return rocsparse_direction_row;
    case HIPSPARSE_DIRECTION_COLUMN:
        return rocsparse_direction_column;
    default:
        throw "Non existent hipsparseDirection_t";
    }
}

hipsparseDirection_t HCCDirectionToHIPDirection(rocsparse_direction_ op)
{
    switch(op)
    {
    case rocsparse_direction_row:
        return HIPSPARSE_DIRECTION_ROW;
    case rocsparse_direction_column:
        return HIPSPARSE_DIRECTION_COLUMN;
    default:
        throw "Non existent rocsparse_direction";
    }
}

rocsparse_order_ hipOrderToHCCOrder(hipsparseOrder_t op)
{
    switch(op)
    {
    case HIPSPARSE_ORDER_ROW:
        return rocsparse_order_row;
    case HIPSPARSE_ORDER_COLUMN:
        return rocsparse_order_column;
    default:
        throw "Non existent hipsparseOrder_t";
    }
}

hipsparseOrder_t HCCOrderToHIPOrder(rocsparse_order_ op)
{
    switch(op)
    {
    case rocsparse_order_row:
        return HIPSPARSE_ORDER_ROW;
    case rocsparse_order_column:
        return HIPSPARSE_ORDER_COLUMN;
    default:
        throw "Non existent rocsparse_order";
    }
}

rocsparse_indextype_ hipIndexTypeToHCCIndexType(hipsparseIndexType_t indextype)
{
    switch(indextype)
    {
    case HIPSPARSE_INDEX_32I:
        return rocsparse_indextype_i32;
    case HIPSPARSE_INDEX_64I:
        return rocsparse_indextype_i64;
    default:
        throw "Non existent hipsparseIndexType_t";
    }
}

hipsparseIndexType_t HCCIndexTypeToHIPIndexType(rocsparse_indextype_ indextype)
{
    switch(indextype)
    {
    case rocsparse_indextype_i32:
        return HIPSPARSE_INDEX_32I;
    case rocsparse_indextype_i64:
        return HIPSPARSE_INDEX_64I;
    default:
        throw "Non existent rocsparse_indextype";
    }
}

rocsparse_datatype_ hipDataTypeToHCCDataType(hipDataType datatype)
{
    switch(datatype)
    {
    case HIP_R_32F:
        return rocsparse_datatype_f32_r;
    case HIP_R_64F:
        return rocsparse_datatype_f64_r;
    case HIP_C_32F:
        return rocsparse_datatype_f32_c;
    case HIP_C_64F:
        return rocsparse_datatype_f64_c;
    default:
        throw "Non existent hipDataType";
    }
}

hipDataType HCCDataTypeToHIPDataType(rocsparse_datatype_ datatype)
{
    switch(datatype)
    {
    case rocsparse_datatype_f32_r:
        return HIP_R_32F;
    case rocsparse_datatype_f64_r:
        return HIP_R_64F;
    case rocsparse_datatype_f32_c:
        return HIP_C_32F;
    case rocsparse_datatype_f64_c:
        return HIP_C_64F;
    default:
        throw "Non existent rocsparse_datatype";
    }
}

rocsparse_spmv_alg_ hipSpMVAlgToHCCSpMVAlg(hipsparseSpMVAlg_t alg)
{
    switch(alg)
    {
    case HIPSPARSE_MV_ALG_DEFAULT:
        return rocsparse_spmv_alg_default;
    case HIPSPARSE_COOMV_ALG:
        return rocsparse_spmv_alg_coo;
    case HIPSPARSE_CSRMV_ALG1:
        return rocsparse_spmv_alg_csr_adaptive;
    case HIPSPARSE_CSRMV_ALG2:
        return rocsparse_spmv_alg_csr_stream;
    default:
        throw "Non existent hipsparseSpMVAlg_t";
    }
}

rocsparse_spmm_alg_ hipSpMMAlgToHCCSpMMAlg(hipsparseSpMMAlg_t alg)
{
    switch(alg)
    {
    case HIPSPARSE_SPMM_ALG_DEFAULT:
        return rocsparse_spmm_alg_default;
    case HIPSPARSE_SPMM_COO_ALG1:
        return rocsparse_spmm_alg_coo_atomic;
    case HIPSPARSE_SPMM_COO_ALG2:
        return rocsparse_spmm_alg_coo_segmented;
    case HIPSPARSE_SPMM_CSR_ALG1:
        return rocsparse_spmm_alg_csr;
    default:
        throw "Non existent hipsparseSpMMAlg_t";
    }
}

rocsparse_sparse_to_dense_alg_ hipSpToDnAlgToHCCSpToDnAlg(hipsparseSparseToDenseAlg_t alg)
{
    switch(alg)
    {
    case HIPSPARSE_SPARSETODENSE_ALG_DEFAULT:
        return rocsparse_sparse_to_dense_alg_default;
    default:
        throw "Non existent hipsparseSparseToDenseAlg_t";
    }
}

hipsparseSparseToDenseAlg_t HCCSpToDnAlgToHipSpToDnAlg(rocsparse_sparse_to_dense_alg_ alg)
{
    switch(alg)
    {
    case rocsparse_sparse_to_dense_alg_default:
        return HIPSPARSE_SPARSETODENSE_ALG_DEFAULT;
    default:
        throw "Non existent rocsparse_sparse_to_dense_alg";
    }
}

rocsparse_dense_to_sparse_alg_ hipDnToSpAlgToHCCDnToSpAlg(hipsparseDenseToSparseAlg_t alg)
{
    switch(alg)
    {
    case HIPSPARSE_DENSETOSPARSE_ALG_DEFAULT:
        return rocsparse_dense_to_sparse_alg_default;
    default:
        throw "Non existent hipsparseDenseToSparseAlg_t";
    }
}

hipsparseDenseToSparseAlg_t HCCDnToSpAlgToHipDnToSpAlg(rocsparse_dense_to_sparse_alg_ alg)
{
    switch(alg)
    {
    case rocsparse_dense_to_sparse_alg_default:
        return HIPSPARSE_DENSETOSPARSE_ALG_DEFAULT;
    default:
        throw "Non existent rocsparse_dense_to_sparse_alg";
    }
}

rocsparse_spgemm_alg_ hipSpGEMMAlgToHCCSpGEMMAlg(hipsparseSpGEMMAlg_t alg)
{
    switch(alg)
    {
    case HIPSPARSE_SPGEMM_DEFAULT:
        return rocsparse_spgemm_alg_default;
    default:
        throw "Non existent hipSpGEMMAlg_t";
    }
}

rocsparse_sddmm_alg_ hipSDDMMAlgToHCCSDDMMAlg(hipsparseSDDMMAlg_t alg)
{
    switch(alg)
    {
    case HIPSPARSE_SDDMM_ALG_DEFAULT:
        return rocsparse_sddmm_alg_default;
    default:
        throw "Non existent hipSDDMMAlg_t";
    }
}

rocsparse_format_ hipFormatToHCCFormat(hipsparseFormat_t format)
{
    switch(format)
    {
    case HIPSPARSE_FORMAT_COO:
        return rocsparse_format_coo;
    case HIPSPARSE_FORMAT_COO_AOS:
        return rocsparse_format_coo_aos;
    case HIPSPARSE_FORMAT_CSR:
        return rocsparse_format_csr;
    default:
        throw "Non existent hipsparseFormat_t";
    }
}

hipsparseFormat_t HCCFormatToHIPFormat(rocsparse_format_ format)
{
    switch(format)
    {
    case rocsparse_format_coo:
        return HIPSPARSE_FORMAT_COO;
    case rocsparse_format_coo_aos:
        return HIPSPARSE_FORMAT_COO_AOS;
    case rocsparse_format_csr:
        return HIPSPARSE_FORMAT_CSR;
    default:
        throw "Non existent rocsparse_format";
    }
}

hipsparseStatus_t hipsparseCreate(hipsparseHandle_t* handle)
{
    // Check if handle is valid
    if(handle == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    int               deviceId;
    hipError_t        err;
    hipsparseStatus_t retval = HIPSPARSE_STATUS_SUCCESS;

    err = hipGetDevice(&deviceId);
    if(err == hipSuccess)
    {
        retval = rocSPARSEStatusToHIPStatus(rocsparse_create_handle((rocsparse_handle*)handle));
    }
    return retval;
}

hipsparseStatus_t hipsparseDestroy(hipsparseHandle_t handle)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_destroy_handle((rocsparse_handle)handle));
}

hipsparseStatus_t hipsparseGetVersion(hipsparseHandle_t handle, int* version)
{
    if(handle == nullptr)
    {
        return HIPSPARSE_STATUS_NOT_INITIALIZED;
    }

    *version = hipsparseVersionMajor * 100000 + hipsparseVersionMinor * 100 + hipsparseVersionPatch;

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseGetGitRevision(hipsparseHandle_t handle, char* rev)
{
    // Get hipSPARSE revision
    if(handle == nullptr)
    {
        return HIPSPARSE_STATUS_NOT_INITIALIZED;
    }

    if(rev == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    static constexpr char v[] = TO_STR(hipsparseVersionTweak);

    char hipsparse_rev[64];
    memcpy(hipsparse_rev, v, sizeof(v));

    // Get rocSPARSE revision
    char rocsparse_rev[64];
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_get_git_rev((rocsparse_handle)handle, rocsparse_rev));

    // Get rocSPARSE version
    int rocsparse_ver;
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_get_version((rocsparse_handle)handle, &rocsparse_ver));

    // Combine
    sprintf(rev,
            "%s (rocSPARSE %d.%d.%d-%s)",
            hipsparse_rev,
            rocsparse_ver / 100000,
            rocsparse_ver / 100 % 1000,
            rocsparse_ver % 100,
            rocsparse_rev);

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseSetStream(hipsparseHandle_t handle, hipStream_t streamId)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_set_stream((rocsparse_handle)handle, streamId));
}

hipsparseStatus_t hipsparseGetStream(hipsparseHandle_t handle, hipStream_t* streamId)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_get_stream((rocsparse_handle)handle, streamId));
}

hipsparseStatus_t hipsparseSetPointerMode(hipsparseHandle_t handle, hipsparsePointerMode_t mode)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_set_pointer_mode((rocsparse_handle)handle, hipPtrModeToHCCPtrMode(mode)));
}

hipsparseStatus_t hipsparseGetPointerMode(hipsparseHandle_t handle, hipsparsePointerMode_t* mode)
{
    rocsparse_pointer_mode_ rocsparse_mode;
    rocsparse_status status = rocsparse_get_pointer_mode((rocsparse_handle)handle, &rocsparse_mode);
    *mode                   = HCCPtrModeToHIPPtrMode(rocsparse_mode);
    return rocSPARSEStatusToHIPStatus(status);
}

hipsparseStatus_t hipsparseCreateMatDescr(hipsparseMatDescr_t* descrA)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_create_mat_descr((rocsparse_mat_descr*)descrA));
}

hipsparseStatus_t hipsparseDestroyMatDescr(hipsparseMatDescr_t descrA)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_destroy_mat_descr((rocsparse_mat_descr)descrA));
}

hipsparseStatus_t hipsparseCopyMatDescr(hipsparseMatDescr_t dest, const hipsparseMatDescr_t src)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_copy_mat_descr((rocsparse_mat_descr)dest, (const rocsparse_mat_descr)src));
}

hipsparseStatus_t hipsparseSetMatType(hipsparseMatDescr_t descrA, hipsparseMatrixType_t type)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_set_mat_type((rocsparse_mat_descr)descrA, hipMatTypeToHCCMatType(type)));
}

hipsparseMatrixType_t hipsparseGetMatType(const hipsparseMatDescr_t descrA)
{
    return HCCMatTypeToHIPMatType(rocsparse_get_mat_type((rocsparse_mat_descr)descrA));
}

hipsparseStatus_t hipsparseSetMatFillMode(hipsparseMatDescr_t descrA, hipsparseFillMode_t fillMode)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_set_mat_fill_mode(
        (rocsparse_mat_descr)descrA, hipFillModeToHCCFillMode(fillMode)));
}

hipsparseFillMode_t hipsparseGetMatFillMode(const hipsparseMatDescr_t descrA)
{
    return HCCFillModeToHIPFillMode(rocsparse_get_mat_fill_mode((rocsparse_mat_descr)descrA));
}

hipsparseStatus_t hipsparseSetMatDiagType(hipsparseMatDescr_t descrA, hipsparseDiagType_t diagType)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_set_mat_diag_type(
        (rocsparse_mat_descr)descrA, hipDiagTypeToHCCDiagType(diagType)));
}

hipsparseDiagType_t hipsparseGetMatDiagType(const hipsparseMatDescr_t descrA)
{
    return HCCDiagTypeToHIPDiagType(rocsparse_get_mat_diag_type((rocsparse_mat_descr)descrA));
}

hipsparseStatus_t hipsparseSetMatIndexBase(hipsparseMatDescr_t descrA, hipsparseIndexBase_t base)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_set_mat_index_base((rocsparse_mat_descr)descrA, hipBaseToHCCBase(base)));
}

hipsparseIndexBase_t hipsparseGetMatIndexBase(const hipsparseMatDescr_t descrA)
{
    return HCCBaseToHIPBase(rocsparse_get_mat_index_base((rocsparse_mat_descr)descrA));
}

hipsparseStatus_t hipsparseCreateHybMat(hipsparseHybMat_t* hybA)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_create_hyb_mat((rocsparse_hyb_mat*)hybA));
}

hipsparseStatus_t hipsparseDestroyHybMat(hipsparseHybMat_t hybA)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_destroy_hyb_mat((rocsparse_hyb_mat)hybA));
}

hipsparseStatus_t hipsparseCreateBsrsv2Info(bsrsv2Info_t* info)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_create_mat_info((rocsparse_mat_info*)info));
}

hipsparseStatus_t hipsparseDestroyBsrsv2Info(bsrsv2Info_t info)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_destroy_mat_info((rocsparse_mat_info)info));
}

hipsparseStatus_t hipsparseCreateBsrsm2Info(bsrsm2Info_t* info)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_create_mat_info((rocsparse_mat_info*)info));
}

hipsparseStatus_t hipsparseDestroyBsrsm2Info(bsrsm2Info_t info)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_destroy_mat_info((rocsparse_mat_info)info));
}

hipsparseStatus_t hipsparseCreateCsrsv2Info(csrsv2Info_t* info)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_create_mat_info((rocsparse_mat_info*)info));
}

hipsparseStatus_t hipsparseDestroyCsrsv2Info(csrsv2Info_t info)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_destroy_mat_info((rocsparse_mat_info)info));
}

hipsparseStatus_t hipsparseCreateColorInfo(hipsparseColorInfo_t* info)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_create_mat_info((rocsparse_mat_info*)info));
}

hipsparseStatus_t hipsparseDestroyColorInfo(hipsparseColorInfo_t info)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_destroy_mat_info((rocsparse_mat_info)info));
}

hipsparseStatus_t hipsparseCreateCsrsm2Info(csrsm2Info_t* info)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_create_mat_info((rocsparse_mat_info*)info));
}

hipsparseStatus_t hipsparseDestroyCsrsm2Info(csrsm2Info_t info)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_destroy_mat_info((rocsparse_mat_info)info));
}

hipsparseStatus_t hipsparseCreateBsrilu02Info(bsrilu02Info_t* info)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_create_mat_info((rocsparse_mat_info*)info));
}

hipsparseStatus_t hipsparseDestroyBsrilu02Info(bsrilu02Info_t info)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_destroy_mat_info((rocsparse_mat_info)info));
}

hipsparseStatus_t hipsparseCreateCsrilu02Info(csrilu02Info_t* info)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_create_mat_info((rocsparse_mat_info*)info));
}

hipsparseStatus_t hipsparseDestroyCsrilu02Info(csrilu02Info_t info)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_destroy_mat_info((rocsparse_mat_info)info));
}

hipsparseStatus_t hipsparseCreateBsric02Info(bsric02Info_t* info)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_create_mat_info((rocsparse_mat_info*)info));
}

hipsparseStatus_t hipsparseDestroyBsric02Info(bsric02Info_t info)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_destroy_mat_info((rocsparse_mat_info)info));
}

hipsparseStatus_t hipsparseCreateCsric02Info(csric02Info_t* info)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_create_mat_info((rocsparse_mat_info*)info));
}

hipsparseStatus_t hipsparseDestroyCsric02Info(csric02Info_t info)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_destroy_mat_info((rocsparse_mat_info)info));
}

hipsparseStatus_t hipsparseCreateCsrgemm2Info(csrgemm2Info_t* info)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_create_mat_info((rocsparse_mat_info*)info));
}

hipsparseStatus_t hipsparseDestroyCsrgemm2Info(csrgemm2Info_t info)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_destroy_mat_info((rocsparse_mat_info)info));
}

hipsparseStatus_t hipsparseCreatePruneInfo(pruneInfo_t* info)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_create_mat_info((rocsparse_mat_info*)info));
}

hipsparseStatus_t hipsparseDestroyPruneInfo(pruneInfo_t info)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_destroy_mat_info((rocsparse_mat_info)info));
}

hipsparseStatus_t hipsparseCreateCsru2csrInfo(csru2csrInfo_t* info)
{
    if(info == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    *info = new csru2csrInfo;

    // Initialize permutation array with nullptr
    (*info)->size = 0;
    (*info)->P    = nullptr;

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseDestroyCsru2csrInfo(csru2csrInfo_t info)
{
    // Check if info structure has been created
    if(info != nullptr)
    {
        // Check if permutation array is allocated
        if(info->P != nullptr)
        {
            RETURN_IF_HIP_ERROR(hipFree(info->P));
            info->size = 0;
        }

        delete info;
    }

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseSaxpyi(hipsparseHandle_t    handle,
                                  int                  nnz,
                                  const float*         alpha,
                                  const float*         xVal,
                                  const int*           xInd,
                                  float*               y,
                                  hipsparseIndexBase_t idxBase)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_saxpyi(
        (rocsparse_handle)handle, nnz, alpha, xVal, xInd, y, hipBaseToHCCBase(idxBase)));
}

hipsparseStatus_t hipsparseDaxpyi(hipsparseHandle_t    handle,
                                  int                  nnz,
                                  const double*        alpha,
                                  const double*        xVal,
                                  const int*           xInd,
                                  double*              y,
                                  hipsparseIndexBase_t idxBase)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_daxpyi(
        (rocsparse_handle)handle, nnz, alpha, xVal, xInd, y, hipBaseToHCCBase(idxBase)));
}

hipsparseStatus_t hipsparseCaxpyi(hipsparseHandle_t    handle,
                                  int                  nnz,
                                  const hipComplex*    alpha,
                                  const hipComplex*    xVal,
                                  const int*           xInd,
                                  hipComplex*          y,
                                  hipsparseIndexBase_t idxBase)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_caxpyi((rocsparse_handle)handle,
                                                       nnz,
                                                       (const rocsparse_float_complex*)alpha,
                                                       (const rocsparse_float_complex*)xVal,
                                                       xInd,
                                                       (rocsparse_float_complex*)y,
                                                       hipBaseToHCCBase(idxBase)));
}

hipsparseStatus_t hipsparseZaxpyi(hipsparseHandle_t       handle,
                                  int                     nnz,
                                  const hipDoubleComplex* alpha,
                                  const hipDoubleComplex* xVal,
                                  const int*              xInd,
                                  hipDoubleComplex*       y,
                                  hipsparseIndexBase_t    idxBase)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_zaxpyi((rocsparse_handle)handle,
                                                       nnz,
                                                       (const rocsparse_double_complex*)alpha,
                                                       (const rocsparse_double_complex*)xVal,
                                                       xInd,
                                                       (rocsparse_double_complex*)y,
                                                       hipBaseToHCCBase(idxBase)));
}

hipsparseStatus_t hipsparseSdoti(hipsparseHandle_t    handle,
                                 int                  nnz,
                                 const float*         xVal,
                                 const int*           xInd,
                                 const float*         y,
                                 float*               result,
                                 hipsparseIndexBase_t idxBase)
{
    // Obtain stream, to explicitly sync (cusparse doti is blocking)
    hipStream_t stream;
    RETURN_IF_HIPSPARSE_ERROR(hipsparseGetStream(handle, &stream));

    // Doti
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_sdoti(
        (rocsparse_handle)handle, nnz, xVal, xInd, y, result, hipBaseToHCCBase(idxBase)));

    // Synchronize stream
    RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseDdoti(hipsparseHandle_t    handle,
                                 int                  nnz,
                                 const double*        xVal,
                                 const int*           xInd,
                                 const double*        y,
                                 double*              result,
                                 hipsparseIndexBase_t idxBase)
{
    // Obtain stream, to explicitly sync (cusparse doti is blocking)
    hipStream_t stream;
    RETURN_IF_HIPSPARSE_ERROR(hipsparseGetStream(handle, &stream));

    // Doti
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_ddoti(
        (rocsparse_handle)handle, nnz, xVal, xInd, y, result, hipBaseToHCCBase(idxBase)));

    // Synchronize stream
    RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseCdoti(hipsparseHandle_t    handle,
                                 int                  nnz,
                                 const hipComplex*    xVal,
                                 const int*           xInd,
                                 const hipComplex*    y,
                                 hipComplex*          result,
                                 hipsparseIndexBase_t idxBase)
{
    // Obtain stream, to explicitly sync (cusparse doti is blocking)
    hipStream_t stream;
    RETURN_IF_HIPSPARSE_ERROR(hipsparseGetStream(handle, &stream));

    // Doti
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_cdoti((rocsparse_handle)handle,
                                              nnz,
                                              (const rocsparse_float_complex*)xVal,
                                              xInd,
                                              (const rocsparse_float_complex*)y,
                                              (rocsparse_float_complex*)result,
                                              hipBaseToHCCBase(idxBase)));

    // Synchronize stream
    RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseZdoti(hipsparseHandle_t       handle,
                                 int                     nnz,
                                 const hipDoubleComplex* xVal,
                                 const int*              xInd,
                                 const hipDoubleComplex* y,
                                 hipDoubleComplex*       result,
                                 hipsparseIndexBase_t    idxBase)
{
    // Obtain stream, to explicitly sync (cusparse doti is blocking)
    hipStream_t stream;
    RETURN_IF_HIPSPARSE_ERROR(hipsparseGetStream(handle, &stream));

    // Doti
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_zdoti((rocsparse_handle)handle,
                                              nnz,
                                              (const rocsparse_double_complex*)xVal,
                                              xInd,
                                              (const rocsparse_double_complex*)y,
                                              (rocsparse_double_complex*)result,
                                              hipBaseToHCCBase(idxBase)));

    // Synchronize stream
    RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseCdotci(hipsparseHandle_t    handle,
                                  int                  nnz,
                                  const hipComplex*    xVal,
                                  const int*           xInd,
                                  const hipComplex*    y,
                                  hipComplex*          result,
                                  hipsparseIndexBase_t idxBase)
{
    // Obtain stream, to explicitly sync (cusparse dotci is blocking)
    hipStream_t stream;
    RETURN_IF_HIPSPARSE_ERROR(hipsparseGetStream(handle, &stream));

    // Dotci
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_cdotci((rocsparse_handle)handle,
                                               nnz,
                                               (const rocsparse_float_complex*)xVal,
                                               xInd,
                                               (const rocsparse_float_complex*)y,
                                               (rocsparse_float_complex*)result,
                                               hipBaseToHCCBase(idxBase)));

    // Synchronize stream
    RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseZdotci(hipsparseHandle_t       handle,
                                  int                     nnz,
                                  const hipDoubleComplex* xVal,
                                  const int*              xInd,
                                  const hipDoubleComplex* y,
                                  hipDoubleComplex*       result,
                                  hipsparseIndexBase_t    idxBase)
{
    // Obtain stream, to explicitly sync (cusparse dotci is blocking)
    hipStream_t stream;
    RETURN_IF_HIPSPARSE_ERROR(hipsparseGetStream(handle, &stream));

    // Dotci
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_zdotci((rocsparse_handle)handle,
                                               nnz,
                                               (const rocsparse_double_complex*)xVal,
                                               xInd,
                                               (const rocsparse_double_complex*)y,
                                               (rocsparse_double_complex*)result,
                                               hipBaseToHCCBase(idxBase)));

    // Synchronize stream
    RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseSgthr(hipsparseHandle_t    handle,
                                 int                  nnz,
                                 const float*         y,
                                 float*               xVal,
                                 const int*           xInd,
                                 hipsparseIndexBase_t idxBase)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_sgthr((rocsparse_handle)handle, nnz, y, xVal, xInd, hipBaseToHCCBase(idxBase)));
}

hipsparseStatus_t hipsparseDgthr(hipsparseHandle_t    handle,
                                 int                  nnz,
                                 const double*        y,
                                 double*              xVal,
                                 const int*           xInd,
                                 hipsparseIndexBase_t idxBase)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_dgthr((rocsparse_handle)handle, nnz, y, xVal, xInd, hipBaseToHCCBase(idxBase)));
}

hipsparseStatus_t hipsparseCgthr(hipsparseHandle_t    handle,
                                 int                  nnz,
                                 const hipComplex*    y,
                                 hipComplex*          xVal,
                                 const int*           xInd,
                                 hipsparseIndexBase_t idxBase)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_cgthr((rocsparse_handle)handle,
                                                      nnz,
                                                      (const rocsparse_float_complex*)y,
                                                      (rocsparse_float_complex*)xVal,
                                                      xInd,
                                                      hipBaseToHCCBase(idxBase)));
}

hipsparseStatus_t hipsparseZgthr(hipsparseHandle_t       handle,
                                 int                     nnz,
                                 const hipDoubleComplex* y,
                                 hipDoubleComplex*       xVal,
                                 const int*              xInd,
                                 hipsparseIndexBase_t    idxBase)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_zgthr((rocsparse_handle)handle,
                                                      nnz,
                                                      (const rocsparse_double_complex*)y,
                                                      (rocsparse_double_complex*)xVal,
                                                      xInd,
                                                      hipBaseToHCCBase(idxBase)));
}

hipsparseStatus_t hipsparseSgthrz(hipsparseHandle_t    handle,
                                  int                  nnz,
                                  float*               y,
                                  float*               xVal,
                                  const int*           xInd,
                                  hipsparseIndexBase_t idxBase)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_sgthrz((rocsparse_handle)handle, nnz, y, xVal, xInd, hipBaseToHCCBase(idxBase)));
}

hipsparseStatus_t hipsparseDgthrz(hipsparseHandle_t    handle,
                                  int                  nnz,
                                  double*              y,
                                  double*              xVal,
                                  const int*           xInd,
                                  hipsparseIndexBase_t idxBase)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_dgthrz((rocsparse_handle)handle, nnz, y, xVal, xInd, hipBaseToHCCBase(idxBase)));
}

hipsparseStatus_t hipsparseCgthrz(hipsparseHandle_t    handle,
                                  int                  nnz,
                                  hipComplex*          y,
                                  hipComplex*          xVal,
                                  const int*           xInd,
                                  hipsparseIndexBase_t idxBase)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_cgthrz((rocsparse_handle)handle,
                                                       nnz,
                                                       (rocsparse_float_complex*)y,
                                                       (rocsparse_float_complex*)xVal,
                                                       xInd,
                                                       hipBaseToHCCBase(idxBase)));
}

hipsparseStatus_t hipsparseZgthrz(hipsparseHandle_t    handle,
                                  int                  nnz,
                                  hipDoubleComplex*    y,
                                  hipDoubleComplex*    xVal,
                                  const int*           xInd,
                                  hipsparseIndexBase_t idxBase)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_zgthrz((rocsparse_handle)handle,
                                                       nnz,
                                                       (rocsparse_double_complex*)y,
                                                       (rocsparse_double_complex*)xVal,
                                                       xInd,
                                                       hipBaseToHCCBase(idxBase)));
}

hipsparseStatus_t hipsparseSroti(hipsparseHandle_t    handle,
                                 int                  nnz,
                                 float*               xVal,
                                 const int*           xInd,
                                 float*               y,
                                 const float*         c,
                                 const float*         s,
                                 hipsparseIndexBase_t idxBase)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_sroti(
        (rocsparse_handle)handle, nnz, xVal, xInd, y, c, s, hipBaseToHCCBase(idxBase)));
}

hipsparseStatus_t hipsparseDroti(hipsparseHandle_t    handle,
                                 int                  nnz,
                                 double*              xVal,
                                 const int*           xInd,
                                 double*              y,
                                 const double*        c,
                                 const double*        s,
                                 hipsparseIndexBase_t idxBase)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_droti(
        (rocsparse_handle)handle, nnz, xVal, xInd, y, c, s, hipBaseToHCCBase(idxBase)));
}

hipsparseStatus_t hipsparseSsctr(hipsparseHandle_t    handle,
                                 int                  nnz,
                                 const float*         xVal,
                                 const int*           xInd,
                                 float*               y,
                                 hipsparseIndexBase_t idxBase)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_ssctr((rocsparse_handle)handle, nnz, xVal, xInd, y, hipBaseToHCCBase(idxBase)));
}

hipsparseStatus_t hipsparseDsctr(hipsparseHandle_t    handle,
                                 int                  nnz,
                                 const double*        xVal,
                                 const int*           xInd,
                                 double*              y,
                                 hipsparseIndexBase_t idxBase)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_dsctr((rocsparse_handle)handle, nnz, xVal, xInd, y, hipBaseToHCCBase(idxBase)));
}

hipsparseStatus_t hipsparseCsctr(hipsparseHandle_t    handle,
                                 int                  nnz,
                                 const hipComplex*    xVal,
                                 const int*           xInd,
                                 hipComplex*          y,
                                 hipsparseIndexBase_t idxBase)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_csctr((rocsparse_handle)handle,
                                                      nnz,
                                                      (const rocsparse_float_complex*)xVal,
                                                      xInd,
                                                      (rocsparse_float_complex*)y,
                                                      hipBaseToHCCBase(idxBase)));
}

hipsparseStatus_t hipsparseZsctr(hipsparseHandle_t       handle,
                                 int                     nnz,
                                 const hipDoubleComplex* xVal,
                                 const int*              xInd,
                                 hipDoubleComplex*       y,
                                 hipsparseIndexBase_t    idxBase)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_zsctr((rocsparse_handle)handle,
                                                      nnz,
                                                      (const rocsparse_double_complex*)xVal,
                                                      xInd,
                                                      (rocsparse_double_complex*)y,
                                                      hipBaseToHCCBase(idxBase)));
}

hipsparseStatus_t hipsparseScsrmv(hipsparseHandle_t         handle,
                                  hipsparseOperation_t      transA,
                                  int                       m,
                                  int                       n,
                                  int                       nnz,
                                  const float*              alpha,
                                  const hipsparseMatDescr_t descrA,
                                  const float*              csrSortedValA,
                                  const int*                csrSortedRowPtrA,
                                  const int*                csrSortedColIndA,
                                  const float*              x,
                                  const float*              beta,
                                  float*                    y)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_scsrmv((rocsparse_handle)handle,
                                                       hipOperationToHCCOperation(transA),
                                                       m,
                                                       n,
                                                       nnz,
                                                       alpha,
                                                       (rocsparse_mat_descr)descrA,
                                                       csrSortedValA,
                                                       csrSortedRowPtrA,
                                                       csrSortedColIndA,
                                                       nullptr,
                                                       x,
                                                       beta,
                                                       y));
}

hipsparseStatus_t hipsparseDcsrmv(hipsparseHandle_t         handle,
                                  hipsparseOperation_t      transA,
                                  int                       m,
                                  int                       n,
                                  int                       nnz,
                                  const double*             alpha,
                                  const hipsparseMatDescr_t descrA,
                                  const double*             csrSortedValA,
                                  const int*                csrSortedRowPtrA,
                                  const int*                csrSortedColIndA,
                                  const double*             x,
                                  const double*             beta,
                                  double*                   y)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_dcsrmv((rocsparse_handle)handle,
                                                       hipOperationToHCCOperation(transA),
                                                       m,
                                                       n,
                                                       nnz,
                                                       alpha,
                                                       (rocsparse_mat_descr)descrA,
                                                       csrSortedValA,
                                                       csrSortedRowPtrA,
                                                       csrSortedColIndA,
                                                       nullptr,
                                                       x,
                                                       beta,
                                                       y));
}

hipsparseStatus_t hipsparseCcsrmv(hipsparseHandle_t         handle,
                                  hipsparseOperation_t      transA,
                                  int                       m,
                                  int                       n,
                                  int                       nnz,
                                  const hipComplex*         alpha,
                                  const hipsparseMatDescr_t descrA,
                                  const hipComplex*         csrSortedValA,
                                  const int*                csrSortedRowPtrA,
                                  const int*                csrSortedColIndA,
                                  const hipComplex*         x,
                                  const hipComplex*         beta,
                                  hipComplex*               y)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_ccsrmv((rocsparse_handle)handle,
                         hipOperationToHCCOperation(transA),
                         m,
                         n,
                         nnz,
                         (const rocsparse_float_complex*)alpha,
                         (rocsparse_mat_descr)descrA,
                         (const rocsparse_float_complex*)csrSortedValA,
                         csrSortedRowPtrA,
                         csrSortedColIndA,
                         nullptr,
                         (const rocsparse_float_complex*)x,
                         (const rocsparse_float_complex*)beta,
                         (rocsparse_float_complex*)y));
}

hipsparseStatus_t hipsparseZcsrmv(hipsparseHandle_t         handle,
                                  hipsparseOperation_t      transA,
                                  int                       m,
                                  int                       n,
                                  int                       nnz,
                                  const hipDoubleComplex*   alpha,
                                  const hipsparseMatDescr_t descrA,
                                  const hipDoubleComplex*   csrSortedValA,
                                  const int*                csrSortedRowPtrA,
                                  const int*                csrSortedColIndA,
                                  const hipDoubleComplex*   x,
                                  const hipDoubleComplex*   beta,
                                  hipDoubleComplex*         y)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_zcsrmv((rocsparse_handle)handle,
                         hipOperationToHCCOperation(transA),
                         m,
                         n,
                         nnz,
                         (const rocsparse_double_complex*)alpha,
                         (rocsparse_mat_descr)descrA,
                         (const rocsparse_double_complex*)csrSortedValA,
                         csrSortedRowPtrA,
                         csrSortedColIndA,
                         nullptr,
                         (const rocsparse_double_complex*)x,
                         (const rocsparse_double_complex*)beta,
                         (rocsparse_double_complex*)y));
}

hipsparseStatus_t
    hipsparseXcsrsv2_zeroPivot(hipsparseHandle_t handle, csrsv2Info_t info, int* position)
{
    // Obtain stream, to explicitly sync (cusparse csrsv2_zeropivot is blocking)
    hipStream_t stream;
    RETURN_IF_HIPSPARSE_ERROR(hipsparseGetStream(handle, &stream));

    // csrsv zero pivot
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_csrsv_zero_pivot(
        (rocsparse_handle)handle, nullptr, (rocsparse_mat_info)info, position));

    // Synchronize stream
    RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseScsrsv2_bufferSize(hipsparseHandle_t         handle,
                                              hipsparseOperation_t      transA,
                                              int                       m,
                                              int                       nnz,
                                              const hipsparseMatDescr_t descrA,
                                              float*                    csrSortedValA,
                                              const int*                csrSortedRowPtrA,
                                              const int*                csrSortedColIndA,
                                              csrsv2Info_t              info,
                                              int*                      pBufferSizeInBytes)
{
    if(pBufferSizeInBytes == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    size_t           buffer_size;
    rocsparse_status status;

    status = rocsparse_scsrsv_buffer_size((rocsparse_handle)handle,
                                          hipOperationToHCCOperation(transA),
                                          m,
                                          nnz,
                                          (rocsparse_mat_descr)descrA,
                                          csrSortedValA,
                                          csrSortedRowPtrA,
                                          csrSortedColIndA,
                                          (rocsparse_mat_info)info,
                                          &buffer_size);

    *pBufferSizeInBytes = (int)buffer_size;

    return rocSPARSEStatusToHIPStatus(status);
}

hipsparseStatus_t hipsparseDcsrsv2_bufferSize(hipsparseHandle_t         handle,
                                              hipsparseOperation_t      transA,
                                              int                       m,
                                              int                       nnz,
                                              const hipsparseMatDescr_t descrA,
                                              double*                   csrSortedValA,
                                              const int*                csrSortedRowPtrA,
                                              const int*                csrSortedColIndA,
                                              csrsv2Info_t              info,
                                              int*                      pBufferSizeInBytes)
{
    if(pBufferSizeInBytes == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    size_t           buffer_size;
    rocsparse_status status;

    status = rocsparse_dcsrsv_buffer_size((rocsparse_handle)handle,
                                          hipOperationToHCCOperation(transA),
                                          m,
                                          nnz,
                                          (rocsparse_mat_descr)descrA,
                                          csrSortedValA,
                                          csrSortedRowPtrA,
                                          csrSortedColIndA,
                                          (rocsparse_mat_info)info,
                                          &buffer_size);

    *pBufferSizeInBytes = (int)buffer_size;

    return rocSPARSEStatusToHIPStatus(status);
}

hipsparseStatus_t hipsparseCcsrsv2_bufferSize(hipsparseHandle_t         handle,
                                              hipsparseOperation_t      transA,
                                              int                       m,
                                              int                       nnz,
                                              const hipsparseMatDescr_t descrA,
                                              hipComplex*               csrSortedValA,
                                              const int*                csrSortedRowPtrA,
                                              const int*                csrSortedColIndA,
                                              csrsv2Info_t              info,
                                              int*                      pBufferSizeInBytes)
{
    if(pBufferSizeInBytes == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    size_t           buffer_size;
    rocsparse_status status;

    status = rocsparse_ccsrsv_buffer_size((rocsparse_handle)handle,
                                          hipOperationToHCCOperation(transA),
                                          m,
                                          nnz,
                                          (rocsparse_mat_descr)descrA,
                                          (rocsparse_float_complex*)csrSortedValA,
                                          csrSortedRowPtrA,
                                          csrSortedColIndA,
                                          (rocsparse_mat_info)info,
                                          &buffer_size);

    *pBufferSizeInBytes = (int)buffer_size;

    return rocSPARSEStatusToHIPStatus(status);
}

hipsparseStatus_t hipsparseZcsrsv2_bufferSize(hipsparseHandle_t         handle,
                                              hipsparseOperation_t      transA,
                                              int                       m,
                                              int                       nnz,
                                              const hipsparseMatDescr_t descrA,
                                              hipDoubleComplex*         csrSortedValA,
                                              const int*                csrSortedRowPtrA,
                                              const int*                csrSortedColIndA,
                                              csrsv2Info_t              info,
                                              int*                      pBufferSizeInBytes)
{
    if(pBufferSizeInBytes == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    size_t           buffer_size;
    rocsparse_status status;

    status = rocsparse_zcsrsv_buffer_size((rocsparse_handle)handle,
                                          hipOperationToHCCOperation(transA),
                                          m,
                                          nnz,
                                          (rocsparse_mat_descr)descrA,
                                          (rocsparse_double_complex*)csrSortedValA,
                                          csrSortedRowPtrA,
                                          csrSortedColIndA,
                                          (rocsparse_mat_info)info,
                                          &buffer_size);

    *pBufferSizeInBytes = (int)buffer_size;

    return rocSPARSEStatusToHIPStatus(status);
}

hipsparseStatus_t hipsparseScsrsv2_bufferSizeExt(hipsparseHandle_t         handle,
                                                 hipsparseOperation_t      transA,
                                                 int                       m,
                                                 int                       nnz,
                                                 const hipsparseMatDescr_t descrA,
                                                 float*                    csrSortedValA,
                                                 const int*                csrSortedRowPtrA,
                                                 const int*                csrSortedColIndA,
                                                 csrsv2Info_t              info,
                                                 size_t*                   pBufferSize)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_scsrsv_buffer_size((rocsparse_handle)handle,
                                     hipOperationToHCCOperation(transA),
                                     m,
                                     nnz,
                                     (rocsparse_mat_descr)descrA,
                                     csrSortedValA,
                                     csrSortedRowPtrA,
                                     csrSortedColIndA,
                                     (rocsparse_mat_info)info,
                                     pBufferSize));
}

hipsparseStatus_t hipsparseDcsrsv2_bufferSizeExt(hipsparseHandle_t         handle,
                                                 hipsparseOperation_t      transA,
                                                 int                       m,
                                                 int                       nnz,
                                                 const hipsparseMatDescr_t descrA,
                                                 double*                   csrSortedValA,
                                                 const int*                csrSortedRowPtrA,
                                                 const int*                csrSortedColIndA,
                                                 csrsv2Info_t              info,
                                                 size_t*                   pBufferSize)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_dcsrsv_buffer_size((rocsparse_handle)handle,
                                     hipOperationToHCCOperation(transA),
                                     m,
                                     nnz,
                                     (rocsparse_mat_descr)descrA,
                                     csrSortedValA,
                                     csrSortedRowPtrA,
                                     csrSortedColIndA,
                                     (rocsparse_mat_info)info,
                                     pBufferSize));
}

hipsparseStatus_t hipsparseCcsrsv2_bufferSizeExt(hipsparseHandle_t         handle,
                                                 hipsparseOperation_t      transA,
                                                 int                       m,
                                                 int                       nnz,
                                                 const hipsparseMatDescr_t descrA,
                                                 hipComplex*               csrSortedValA,
                                                 const int*                csrSortedRowPtrA,
                                                 const int*                csrSortedColIndA,
                                                 csrsv2Info_t              info,
                                                 size_t*                   pBufferSize)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_ccsrsv_buffer_size((rocsparse_handle)handle,
                                     hipOperationToHCCOperation(transA),
                                     m,
                                     nnz,
                                     (rocsparse_mat_descr)descrA,
                                     (rocsparse_float_complex*)csrSortedValA,
                                     csrSortedRowPtrA,
                                     csrSortedColIndA,
                                     (rocsparse_mat_info)info,
                                     pBufferSize));
}

hipsparseStatus_t hipsparseZcsrsv2_bufferSizeExt(hipsparseHandle_t         handle,
                                                 hipsparseOperation_t      transA,
                                                 int                       m,
                                                 int                       nnz,
                                                 const hipsparseMatDescr_t descrA,
                                                 hipDoubleComplex*         csrSortedValA,
                                                 const int*                csrSortedRowPtrA,
                                                 const int*                csrSortedColIndA,
                                                 csrsv2Info_t              info,
                                                 size_t*                   pBufferSize)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_zcsrsv_buffer_size((rocsparse_handle)handle,
                                     hipOperationToHCCOperation(transA),
                                     m,
                                     nnz,
                                     (rocsparse_mat_descr)descrA,
                                     (rocsparse_double_complex*)csrSortedValA,
                                     csrSortedRowPtrA,
                                     csrSortedColIndA,
                                     (rocsparse_mat_info)info,
                                     pBufferSize));
}

hipsparseStatus_t hipsparseScsrsv2_analysis(hipsparseHandle_t         handle,
                                            hipsparseOperation_t      transA,
                                            int                       m,
                                            int                       nnz,
                                            const hipsparseMatDescr_t descrA,
                                            const float*              csrSortedValA,
                                            const int*                csrSortedRowPtrA,
                                            const int*                csrSortedColIndA,
                                            csrsv2Info_t              info,
                                            hipsparseSolvePolicy_t    policy,
                                            void*                     pBuffer)
{
    // Obtain stream, to explicitly sync (cusparse csrsv2_analysis is blocking)
    hipStream_t stream;
    RETURN_IF_HIPSPARSE_ERROR(hipsparseGetStream(handle, &stream));

    // csrsv analysis
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_scsrsv_analysis((rocsparse_handle)handle,
                                                        hipOperationToHCCOperation(transA),
                                                        m,
                                                        nnz,
                                                        (rocsparse_mat_descr)descrA,
                                                        csrSortedValA,
                                                        csrSortedRowPtrA,
                                                        csrSortedColIndA,
                                                        (rocsparse_mat_info)info,
                                                        rocsparse_analysis_policy_force,
                                                        rocsparse_solve_policy_auto,
                                                        pBuffer));

    // Synchronize stream
    RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseDcsrsv2_analysis(hipsparseHandle_t         handle,
                                            hipsparseOperation_t      transA,
                                            int                       m,
                                            int                       nnz,
                                            const hipsparseMatDescr_t descrA,
                                            const double*             csrSortedValA,
                                            const int*                csrSortedRowPtrA,
                                            const int*                csrSortedColIndA,
                                            csrsv2Info_t              info,
                                            hipsparseSolvePolicy_t    policy,
                                            void*                     pBuffer)
{
    // Obtain stream, to explicitly sync (cusparse csrsv2_analysis is blocking)
    hipStream_t stream;
    RETURN_IF_HIPSPARSE_ERROR(hipsparseGetStream(handle, &stream));

    // csrsv analysis
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_dcsrsv_analysis((rocsparse_handle)handle,
                                                        hipOperationToHCCOperation(transA),
                                                        m,
                                                        nnz,
                                                        (rocsparse_mat_descr)descrA,
                                                        csrSortedValA,
                                                        csrSortedRowPtrA,
                                                        csrSortedColIndA,
                                                        (rocsparse_mat_info)info,
                                                        rocsparse_analysis_policy_force,
                                                        rocsparse_solve_policy_auto,
                                                        pBuffer));

    // Synchronize stream
    RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseCcsrsv2_analysis(hipsparseHandle_t         handle,
                                            hipsparseOperation_t      transA,
                                            int                       m,
                                            int                       nnz,
                                            const hipsparseMatDescr_t descrA,
                                            const hipComplex*         csrSortedValA,
                                            const int*                csrSortedRowPtrA,
                                            const int*                csrSortedColIndA,
                                            csrsv2Info_t              info,
                                            hipsparseSolvePolicy_t    policy,
                                            void*                     pBuffer)
{
    // Obtain stream, to explicitly sync (cusparse csrsv2_analysis is blocking)
    hipStream_t stream;
    RETURN_IF_HIPSPARSE_ERROR(hipsparseGetStream(handle, &stream));

    // csrsv analysis
    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse_ccsrsv_analysis((rocsparse_handle)handle,
                                  hipOperationToHCCOperation(transA),
                                  m,
                                  nnz,
                                  (rocsparse_mat_descr)descrA,
                                  (const rocsparse_float_complex*)csrSortedValA,
                                  csrSortedRowPtrA,
                                  csrSortedColIndA,
                                  (rocsparse_mat_info)info,
                                  rocsparse_analysis_policy_force,
                                  rocsparse_solve_policy_auto,
                                  pBuffer));

    // Synchronize stream
    RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseZcsrsv2_analysis(hipsparseHandle_t         handle,
                                            hipsparseOperation_t      transA,
                                            int                       m,
                                            int                       nnz,
                                            const hipsparseMatDescr_t descrA,
                                            const hipDoubleComplex*   csrSortedValA,
                                            const int*                csrSortedRowPtrA,
                                            const int*                csrSortedColIndA,
                                            csrsv2Info_t              info,
                                            hipsparseSolvePolicy_t    policy,
                                            void*                     pBuffer)
{
    // Obtain stream, to explicitly sync (cusparse csrsv2_analysis is blocking)
    hipStream_t stream;
    RETURN_IF_HIPSPARSE_ERROR(hipsparseGetStream(handle, &stream));

    // csrsv analysis
    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse_zcsrsv_analysis((rocsparse_handle)handle,
                                  hipOperationToHCCOperation(transA),
                                  m,
                                  nnz,
                                  (rocsparse_mat_descr)descrA,
                                  (const rocsparse_double_complex*)csrSortedValA,
                                  csrSortedRowPtrA,
                                  csrSortedColIndA,
                                  (rocsparse_mat_info)info,
                                  rocsparse_analysis_policy_force,
                                  rocsparse_solve_policy_auto,
                                  pBuffer));

    // Synchronize stream
    RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseScsrsv2_solve(hipsparseHandle_t         handle,
                                         hipsparseOperation_t      transA,
                                         int                       m,
                                         int                       nnz,
                                         const float*              alpha,
                                         const hipsparseMatDescr_t descrA,
                                         const float*              csrSortedValA,
                                         const int*                csrSortedRowPtrA,
                                         const int*                csrSortedColIndA,
                                         csrsv2Info_t              info,
                                         const float*              f,
                                         float*                    x,
                                         hipsparseSolvePolicy_t    policy,
                                         void*                     pBuffer)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_scsrsv_solve((rocsparse_handle)handle,
                                                             hipOperationToHCCOperation(transA),
                                                             m,
                                                             nnz,
                                                             alpha,
                                                             (rocsparse_mat_descr)descrA,
                                                             csrSortedValA,
                                                             csrSortedRowPtrA,
                                                             csrSortedColIndA,
                                                             (rocsparse_mat_info)info,
                                                             f,
                                                             x,
                                                             rocsparse_solve_policy_auto,
                                                             pBuffer));
}

hipsparseStatus_t hipsparseDcsrsv2_solve(hipsparseHandle_t         handle,
                                         hipsparseOperation_t      transA,
                                         int                       m,
                                         int                       nnz,
                                         const double*             alpha,
                                         const hipsparseMatDescr_t descrA,
                                         const double*             csrSortedValA,
                                         const int*                csrSortedRowPtrA,
                                         const int*                csrSortedColIndA,
                                         csrsv2Info_t              info,
                                         const double*             f,
                                         double*                   x,
                                         hipsparseSolvePolicy_t    policy,
                                         void*                     pBuffer)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_dcsrsv_solve((rocsparse_handle)handle,
                                                             hipOperationToHCCOperation(transA),
                                                             m,
                                                             nnz,
                                                             alpha,
                                                             (rocsparse_mat_descr)descrA,
                                                             csrSortedValA,
                                                             csrSortedRowPtrA,
                                                             csrSortedColIndA,
                                                             (rocsparse_mat_info)info,
                                                             f,
                                                             x,
                                                             rocsparse_solve_policy_auto,
                                                             pBuffer));
}

hipsparseStatus_t hipsparseCcsrsv2_solve(hipsparseHandle_t         handle,
                                         hipsparseOperation_t      transA,
                                         int                       m,
                                         int                       nnz,
                                         const hipComplex*         alpha,
                                         const hipsparseMatDescr_t descrA,
                                         const hipComplex*         csrSortedValA,
                                         const int*                csrSortedRowPtrA,
                                         const int*                csrSortedColIndA,
                                         csrsv2Info_t              info,
                                         const hipComplex*         f,
                                         hipComplex*               x,
                                         hipsparseSolvePolicy_t    policy,
                                         void*                     pBuffer)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_ccsrsv_solve((rocsparse_handle)handle,
                               hipOperationToHCCOperation(transA),
                               m,
                               nnz,
                               (const rocsparse_float_complex*)alpha,
                               (rocsparse_mat_descr)descrA,
                               (const rocsparse_float_complex*)csrSortedValA,
                               csrSortedRowPtrA,
                               csrSortedColIndA,
                               (rocsparse_mat_info)info,
                               (const rocsparse_float_complex*)f,
                               (rocsparse_float_complex*)x,
                               rocsparse_solve_policy_auto,
                               pBuffer));
}

hipsparseStatus_t hipsparseZcsrsv2_solve(hipsparseHandle_t         handle,
                                         hipsparseOperation_t      transA,
                                         int                       m,
                                         int                       nnz,
                                         const hipDoubleComplex*   alpha,
                                         const hipsparseMatDescr_t descrA,
                                         const hipDoubleComplex*   csrSortedValA,
                                         const int*                csrSortedRowPtrA,
                                         const int*                csrSortedColIndA,
                                         csrsv2Info_t              info,
                                         const hipDoubleComplex*   f,
                                         hipDoubleComplex*         x,
                                         hipsparseSolvePolicy_t    policy,
                                         void*                     pBuffer)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_zcsrsv_solve((rocsparse_handle)handle,
                               hipOperationToHCCOperation(transA),
                               m,
                               nnz,
                               (const rocsparse_double_complex*)alpha,
                               (rocsparse_mat_descr)descrA,
                               (const rocsparse_double_complex*)csrSortedValA,
                               csrSortedRowPtrA,
                               csrSortedColIndA,
                               (rocsparse_mat_info)info,
                               (const rocsparse_double_complex*)f,
                               (rocsparse_double_complex*)x,
                               rocsparse_solve_policy_auto,
                               pBuffer));
}

hipsparseStatus_t hipsparseShybmv(hipsparseHandle_t         handle,
                                  hipsparseOperation_t      transA,
                                  const float*              alpha,
                                  const hipsparseMatDescr_t descrA,
                                  const hipsparseHybMat_t   hybA,
                                  const float*              x,
                                  const float*              beta,
                                  float*                    y)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_shybmv((rocsparse_handle)handle,
                                                       hipOperationToHCCOperation(transA),
                                                       alpha,
                                                       (rocsparse_mat_descr)descrA,
                                                       (rocsparse_hyb_mat)hybA,
                                                       x,
                                                       beta,
                                                       y));
}

hipsparseStatus_t hipsparseDhybmv(hipsparseHandle_t         handle,
                                  hipsparseOperation_t      transA,
                                  const double*             alpha,
                                  const hipsparseMatDescr_t descrA,
                                  const hipsparseHybMat_t   hybA,
                                  const double*             x,
                                  const double*             beta,
                                  double*                   y)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_dhybmv((rocsparse_handle)handle,
                                                       hipOperationToHCCOperation(transA),
                                                       alpha,
                                                       (rocsparse_mat_descr)descrA,
                                                       (rocsparse_hyb_mat)hybA,
                                                       x,
                                                       beta,
                                                       y));
}

hipsparseStatus_t hipsparseChybmv(hipsparseHandle_t         handle,
                                  hipsparseOperation_t      transA,
                                  const hipComplex*         alpha,
                                  const hipsparseMatDescr_t descrA,
                                  const hipsparseHybMat_t   hybA,
                                  const hipComplex*         x,
                                  const hipComplex*         beta,
                                  hipComplex*               y)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_chybmv((rocsparse_handle)handle,
                                                       hipOperationToHCCOperation(transA),
                                                       (const rocsparse_float_complex*)alpha,
                                                       (rocsparse_mat_descr)descrA,
                                                       (rocsparse_hyb_mat)hybA,
                                                       (const rocsparse_float_complex*)x,
                                                       (const rocsparse_float_complex*)beta,
                                                       (rocsparse_float_complex*)y));
}

hipsparseStatus_t hipsparseZhybmv(hipsparseHandle_t         handle,
                                  hipsparseOperation_t      transA,
                                  const hipDoubleComplex*   alpha,
                                  const hipsparseMatDescr_t descrA,
                                  const hipsparseHybMat_t   hybA,
                                  const hipDoubleComplex*   x,
                                  const hipDoubleComplex*   beta,
                                  hipDoubleComplex*         y)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_zhybmv((rocsparse_handle)handle,
                                                       hipOperationToHCCOperation(transA),
                                                       (const rocsparse_double_complex*)alpha,
                                                       (rocsparse_mat_descr)descrA,
                                                       (rocsparse_hyb_mat)hybA,
                                                       (const rocsparse_double_complex*)x,
                                                       (const rocsparse_double_complex*)beta,
                                                       (rocsparse_double_complex*)y));
}

hipsparseStatus_t hipsparseSbsrmv(hipsparseHandle_t         handle,
                                  hipsparseDirection_t      dirA,
                                  hipsparseOperation_t      transA,
                                  int                       mb,
                                  int                       nb,
                                  int                       nnzb,
                                  const float*              alpha,
                                  const hipsparseMatDescr_t descrA,
                                  const float*              bsrSortedValA,
                                  const int*                bsrSortedRowPtrA,
                                  const int*                bsrSortedColIndA,
                                  int                       blockDim,
                                  const float*              x,
                                  const float*              beta,
                                  float*                    y)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_sbsrmv((rocsparse_handle)handle,
                                                       hipDirectionToHCCDirection(dirA),
                                                       hipOperationToHCCOperation(transA),
                                                       mb,
                                                       nb,
                                                       nnzb,
                                                       alpha,
                                                       (const rocsparse_mat_descr)descrA,
                                                       bsrSortedValA,
                                                       bsrSortedRowPtrA,
                                                       bsrSortedColIndA,
                                                       blockDim,
                                                       x,
                                                       beta,
                                                       y));
}

hipsparseStatus_t hipsparseDbsrmv(hipsparseHandle_t         handle,
                                  hipsparseDirection_t      dirA,
                                  hipsparseOperation_t      transA,
                                  int                       mb,
                                  int                       nb,
                                  int                       nnzb,
                                  const double*             alpha,
                                  const hipsparseMatDescr_t descrA,
                                  const double*             bsrSortedValA,
                                  const int*                bsrSortedRowPtrA,
                                  const int*                bsrSortedColIndA,
                                  int                       blockDim,
                                  const double*             x,
                                  const double*             beta,
                                  double*                   y)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_dbsrmv((rocsparse_handle)handle,
                                                       hipDirectionToHCCDirection(dirA),
                                                       hipOperationToHCCOperation(transA),
                                                       mb,
                                                       nb,
                                                       nnzb,
                                                       alpha,
                                                       (const rocsparse_mat_descr)descrA,
                                                       bsrSortedValA,
                                                       bsrSortedRowPtrA,
                                                       bsrSortedColIndA,
                                                       blockDim,
                                                       x,
                                                       beta,
                                                       y));
}

hipsparseStatus_t hipsparseCbsrmv(hipsparseHandle_t         handle,
                                  hipsparseDirection_t      dirA,
                                  hipsparseOperation_t      transA,
                                  int                       mb,
                                  int                       nb,
                                  int                       nnzb,
                                  const hipComplex*         alpha,
                                  const hipsparseMatDescr_t descrA,
                                  const hipComplex*         bsrSortedValA,
                                  const int*                bsrSortedRowPtrA,
                                  const int*                bsrSortedColIndA,
                                  int                       blockDim,
                                  const hipComplex*         x,
                                  const hipComplex*         beta,
                                  hipComplex*               y)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_cbsrmv((rocsparse_handle)handle,
                         hipDirectionToHCCDirection(dirA),
                         hipOperationToHCCOperation(transA),
                         mb,
                         nb,
                         nnzb,
                         (const rocsparse_float_complex*)alpha,
                         (const rocsparse_mat_descr)descrA,
                         (const rocsparse_float_complex*)bsrSortedValA,
                         bsrSortedRowPtrA,
                         bsrSortedColIndA,
                         blockDim,
                         (const rocsparse_float_complex*)x,
                         (const rocsparse_float_complex*)beta,
                         (rocsparse_float_complex*)y));
}

hipsparseStatus_t hipsparseZbsrmv(hipsparseHandle_t         handle,
                                  hipsparseDirection_t      dirA,
                                  hipsparseOperation_t      transA,
                                  int                       mb,
                                  int                       nb,
                                  int                       nnzb,
                                  const hipDoubleComplex*   alpha,
                                  const hipsparseMatDescr_t descrA,
                                  const hipDoubleComplex*   bsrSortedValA,
                                  const int*                bsrSortedRowPtrA,
                                  const int*                bsrSortedColIndA,
                                  int                       blockDim,
                                  const hipDoubleComplex*   x,
                                  const hipDoubleComplex*   beta,
                                  hipDoubleComplex*         y)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_zbsrmv((rocsparse_handle)handle,
                         hipDirectionToHCCDirection(dirA),
                         hipOperationToHCCOperation(transA),
                         mb,
                         nb,
                         nnzb,
                         (const rocsparse_double_complex*)alpha,
                         (const rocsparse_mat_descr)descrA,
                         (const rocsparse_double_complex*)bsrSortedValA,
                         bsrSortedRowPtrA,
                         bsrSortedColIndA,
                         blockDim,
                         (const rocsparse_double_complex*)x,
                         (const rocsparse_double_complex*)beta,
                         (rocsparse_double_complex*)y));
}

hipsparseStatus_t hipsparseSbsrxmv(hipsparseHandle_t         handle,
                                   hipsparseDirection_t      dir,
                                   hipsparseOperation_t      trans,
                                   int                       sizeOfMask,
                                   int                       mb,
                                   int                       nb,
                                   int                       nnzb,
                                   const float*              alpha,
                                   const hipsparseMatDescr_t descr,
                                   const float*              bsrVal,
                                   const int*                bsrMaskPtr,
                                   const int*                bsrRowPtr,
                                   const int*                bsrEndPtr,
                                   const int*                bsrColInd,
                                   int                       blockDim,
                                   const float*              x,
                                   const float*              beta,
                                   float*                    y)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_sbsrxmv((rocsparse_handle)handle,
                                                        hipDirectionToHCCDirection(dir),
                                                        hipOperationToHCCOperation(trans),
                                                        sizeOfMask,
                                                        mb,
                                                        nb,
                                                        nnzb,
                                                        alpha,
                                                        (const rocsparse_mat_descr)descr,
                                                        bsrVal,
                                                        bsrMaskPtr,
                                                        bsrRowPtr,
                                                        bsrEndPtr,
                                                        bsrColInd,
                                                        blockDim,
                                                        x,
                                                        beta,
                                                        y));
}

hipsparseStatus_t hipsparseDbsrxmv(hipsparseHandle_t         handle,
                                   hipsparseDirection_t      dir,
                                   hipsparseOperation_t      trans,
                                   int                       sizeOfMask,
                                   int                       mb,
                                   int                       nb,
                                   int                       nnzb,
                                   const double*             alpha,
                                   const hipsparseMatDescr_t descr,
                                   const double*             bsrVal,
                                   const int*                bsrMaskPtr,
                                   const int*                bsrRowPtr,
                                   const int*                bsrEndPtr,
                                   const int*                bsrColInd,
                                   int                       blockDim,
                                   const double*             x,
                                   const double*             beta,
                                   double*                   y)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_dbsrxmv((rocsparse_handle)handle,
                                                        hipDirectionToHCCDirection(dir),
                                                        hipOperationToHCCOperation(trans),
                                                        sizeOfMask,
                                                        mb,
                                                        nb,
                                                        nnzb,
                                                        alpha,
                                                        (const rocsparse_mat_descr)descr,
                                                        bsrVal,
                                                        bsrMaskPtr,
                                                        bsrRowPtr,
                                                        bsrEndPtr,
                                                        bsrColInd,
                                                        blockDim,
                                                        x,
                                                        beta,
                                                        y));
}

hipsparseStatus_t hipsparseCbsrxmv(hipsparseHandle_t         handle,
                                   hipsparseDirection_t      dir,
                                   hipsparseOperation_t      trans,
                                   int                       sizeOfMask,
                                   int                       mb,
                                   int                       nb,
                                   int                       nnzb,
                                   const hipComplex*         alpha,
                                   const hipsparseMatDescr_t descr,
                                   const hipComplex*         bsrVal,
                                   const int*                bsrMaskPtr,
                                   const int*                bsrRowPtr,
                                   const int*                bsrEndPtr,
                                   const int*                bsrColInd,
                                   int                       blockDim,
                                   const hipComplex*         x,
                                   const hipComplex*         beta,
                                   hipComplex*               y)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_cbsrxmv((rocsparse_handle)handle,
                                                        hipDirectionToHCCDirection(dir),
                                                        hipOperationToHCCOperation(trans),
                                                        sizeOfMask,
                                                        mb,
                                                        nb,
                                                        nnzb,
                                                        (const rocsparse_float_complex*)alpha,
                                                        (const rocsparse_mat_descr)descr,
                                                        (const rocsparse_float_complex*)bsrVal,
                                                        bsrMaskPtr,
                                                        bsrRowPtr,
                                                        bsrEndPtr,
                                                        bsrColInd,
                                                        blockDim,
                                                        (const rocsparse_float_complex*)x,
                                                        (const rocsparse_float_complex*)beta,
                                                        (rocsparse_float_complex*)y));
}

hipsparseStatus_t hipsparseZbsrxmv(hipsparseHandle_t         handle,
                                   hipsparseDirection_t      dir,
                                   hipsparseOperation_t      trans,
                                   int                       sizeOfMask,
                                   int                       mb,
                                   int                       nb,
                                   int                       nnzb,
                                   const hipDoubleComplex*   alpha,
                                   const hipsparseMatDescr_t descr,
                                   const hipDoubleComplex*   bsrVal,
                                   const int*                bsrMaskPtr,
                                   const int*                bsrRowPtr,
                                   const int*                bsrEndPtr,
                                   const int*                bsrColInd,
                                   int                       blockDim,
                                   const hipDoubleComplex*   x,
                                   const hipDoubleComplex*   beta,
                                   hipDoubleComplex*         y)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_zbsrxmv((rocsparse_handle)handle,
                                                        hipDirectionToHCCDirection(dir),
                                                        hipOperationToHCCOperation(trans),
                                                        sizeOfMask,
                                                        mb,
                                                        nb,
                                                        nnzb,
                                                        (const rocsparse_double_complex*)alpha,
                                                        (const rocsparse_mat_descr)descr,
                                                        (const rocsparse_double_complex*)bsrVal,
                                                        bsrMaskPtr,
                                                        bsrRowPtr,
                                                        bsrEndPtr,
                                                        bsrColInd,
                                                        blockDim,
                                                        (const rocsparse_double_complex*)x,
                                                        (const rocsparse_double_complex*)beta,
                                                        (rocsparse_double_complex*)y));
}

hipsparseStatus_t
    hipsparseXbsrsv2_zeroPivot(hipsparseHandle_t handle, bsrsv2Info_t info, int* position)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_bsrsv_zero_pivot((rocsparse_handle)handle, (rocsparse_mat_info)info, position));
}

hipsparseStatus_t hipsparseSbsrsv2_bufferSize(hipsparseHandle_t         handle,
                                              hipsparseDirection_t      dir,
                                              hipsparseOperation_t      transA,
                                              int                       mb,
                                              int                       nnzb,
                                              const hipsparseMatDescr_t descrA,
                                              float*                    bsrSortedValA,
                                              const int*                bsrSortedRowPtrA,
                                              const int*                bsrSortedColIndA,
                                              int                       blockDim,
                                              bsrsv2Info_t              info,
                                              int*                      pBufferSizeInBytes)
{
    if(pBufferSizeInBytes == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    size_t           buffer_size;
    rocsparse_status status;

    status = rocsparse_sbsrsv_buffer_size((rocsparse_handle)handle,
                                          hipDirectionToHCCDirection(dir),
                                          hipOperationToHCCOperation(transA),
                                          mb,
                                          nnzb,
                                          (rocsparse_mat_descr)descrA,
                                          bsrSortedValA,
                                          bsrSortedRowPtrA,
                                          bsrSortedColIndA,
                                          blockDim,
                                          (rocsparse_mat_info)info,
                                          &buffer_size);

    *pBufferSizeInBytes = (int)buffer_size;

    return rocSPARSEStatusToHIPStatus(status);
}

hipsparseStatus_t hipsparseDbsrsv2_bufferSize(hipsparseHandle_t         handle,
                                              hipsparseDirection_t      dir,
                                              hipsparseOperation_t      transA,
                                              int                       mb,
                                              int                       nnzb,
                                              const hipsparseMatDescr_t descrA,
                                              double*                   bsrSortedValA,
                                              const int*                bsrSortedRowPtrA,
                                              const int*                bsrSortedColIndA,
                                              int                       blockDim,
                                              bsrsv2Info_t              info,
                                              int*                      pBufferSizeInBytes)
{
    if(pBufferSizeInBytes == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    size_t           buffer_size;
    rocsparse_status status;

    status = rocsparse_dbsrsv_buffer_size((rocsparse_handle)handle,
                                          hipDirectionToHCCDirection(dir),
                                          hipOperationToHCCOperation(transA),
                                          mb,
                                          nnzb,
                                          (rocsparse_mat_descr)descrA,
                                          bsrSortedValA,
                                          bsrSortedRowPtrA,
                                          bsrSortedColIndA,
                                          blockDim,
                                          (rocsparse_mat_info)info,
                                          &buffer_size);

    *pBufferSizeInBytes = (int)buffer_size;

    return rocSPARSEStatusToHIPStatus(status);
}

hipsparseStatus_t hipsparseCbsrsv2_bufferSize(hipsparseHandle_t         handle,
                                              hipsparseDirection_t      dir,
                                              hipsparseOperation_t      transA,
                                              int                       mb,
                                              int                       nnzb,
                                              const hipsparseMatDescr_t descrA,
                                              hipComplex*               bsrSortedValA,
                                              const int*                bsrSortedRowPtrA,
                                              const int*                bsrSortedColIndA,
                                              int                       blockDim,
                                              bsrsv2Info_t              info,
                                              int*                      pBufferSizeInBytes)
{
    if(pBufferSizeInBytes == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    size_t           buffer_size;
    rocsparse_status status;

    status = rocsparse_cbsrsv_buffer_size((rocsparse_handle)handle,
                                          hipDirectionToHCCDirection(dir),
                                          hipOperationToHCCOperation(transA),
                                          mb,
                                          nnzb,
                                          (rocsparse_mat_descr)descrA,
                                          (rocsparse_float_complex*)bsrSortedValA,
                                          bsrSortedRowPtrA,
                                          bsrSortedColIndA,
                                          blockDim,
                                          (rocsparse_mat_info)info,
                                          &buffer_size);

    *pBufferSizeInBytes = (int)buffer_size;

    return rocSPARSEStatusToHIPStatus(status);
}

hipsparseStatus_t hipsparseZbsrsv2_bufferSize(hipsparseHandle_t         handle,
                                              hipsparseDirection_t      dir,
                                              hipsparseOperation_t      transA,
                                              int                       mb,
                                              int                       nnzb,
                                              const hipsparseMatDescr_t descrA,
                                              hipDoubleComplex*         bsrSortedValA,
                                              const int*                bsrSortedRowPtrA,
                                              const int*                bsrSortedColIndA,
                                              int                       blockDim,
                                              bsrsv2Info_t              info,
                                              int*                      pBufferSizeInBytes)
{
    if(pBufferSizeInBytes == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    size_t           buffer_size;
    rocsparse_status status;

    status = rocsparse_zbsrsv_buffer_size((rocsparse_handle)handle,
                                          hipDirectionToHCCDirection(dir),
                                          hipOperationToHCCOperation(transA),
                                          mb,
                                          nnzb,
                                          (rocsparse_mat_descr)descrA,
                                          (rocsparse_double_complex*)bsrSortedValA,
                                          bsrSortedRowPtrA,
                                          bsrSortedColIndA,
                                          blockDim,
                                          (rocsparse_mat_info)info,
                                          &buffer_size);

    *pBufferSizeInBytes = (int)buffer_size;

    return rocSPARSEStatusToHIPStatus(status);
}

hipsparseStatus_t hipsparseSbsrsv2_bufferSizeExt(hipsparseHandle_t         handle,
                                                 hipsparseDirection_t      dir,
                                                 hipsparseOperation_t      transA,
                                                 int                       mb,
                                                 int                       nnzb,
                                                 const hipsparseMatDescr_t descrA,
                                                 float*                    bsrSortedValA,
                                                 const int*                bsrSortedRowPtrA,
                                                 const int*                bsrSortedColIndA,
                                                 int                       blockDim,
                                                 bsrsv2Info_t              info,
                                                 size_t*                   pBufferSize)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_sbsrsv_buffer_size((rocsparse_handle)handle,
                                     hipDirectionToHCCDirection(dir),
                                     hipOperationToHCCOperation(transA),
                                     mb,
                                     nnzb,
                                     (rocsparse_mat_descr)descrA,
                                     bsrSortedValA,
                                     bsrSortedRowPtrA,
                                     bsrSortedColIndA,
                                     blockDim,
                                     (rocsparse_mat_info)info,
                                     pBufferSize));
}

hipsparseStatus_t hipsparseDbsrsv2_bufferSizeExt(hipsparseHandle_t         handle,
                                                 hipsparseDirection_t      dir,
                                                 hipsparseOperation_t      transA,
                                                 int                       mb,
                                                 int                       nnzb,
                                                 const hipsparseMatDescr_t descrA,
                                                 double*                   bsrSortedValA,
                                                 const int*                bsrSortedRowPtrA,
                                                 const int*                bsrSortedColIndA,
                                                 int                       blockDim,
                                                 bsrsv2Info_t              info,
                                                 size_t*                   pBufferSize)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_dbsrsv_buffer_size((rocsparse_handle)handle,
                                     hipDirectionToHCCDirection(dir),
                                     hipOperationToHCCOperation(transA),
                                     mb,
                                     nnzb,
                                     (rocsparse_mat_descr)descrA,
                                     bsrSortedValA,
                                     bsrSortedRowPtrA,
                                     bsrSortedColIndA,
                                     blockDim,
                                     (rocsparse_mat_info)info,
                                     pBufferSize));
}

hipsparseStatus_t hipsparseCbsrsv2_bufferSizeExt(hipsparseHandle_t         handle,
                                                 hipsparseDirection_t      dir,
                                                 hipsparseOperation_t      transA,
                                                 int                       mb,
                                                 int                       nnzb,
                                                 const hipsparseMatDescr_t descrA,
                                                 hipComplex*               bsrSortedValA,
                                                 const int*                bsrSortedRowPtrA,
                                                 const int*                bsrSortedColIndA,
                                                 int                       blockDim,
                                                 bsrsv2Info_t              info,
                                                 size_t*                   pBufferSize)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_cbsrsv_buffer_size((rocsparse_handle)handle,
                                     hipDirectionToHCCDirection(dir),
                                     hipOperationToHCCOperation(transA),
                                     mb,
                                     nnzb,
                                     (rocsparse_mat_descr)descrA,
                                     (rocsparse_float_complex*)bsrSortedValA,
                                     bsrSortedRowPtrA,
                                     bsrSortedColIndA,
                                     blockDim,
                                     (rocsparse_mat_info)info,
                                     pBufferSize));
}

hipsparseStatus_t hipsparseZbsrsv2_bufferSizeExt(hipsparseHandle_t         handle,
                                                 hipsparseDirection_t      dir,
                                                 hipsparseOperation_t      transA,
                                                 int                       mb,
                                                 int                       nnzb,
                                                 const hipsparseMatDescr_t descrA,
                                                 hipDoubleComplex*         bsrSortedValA,
                                                 const int*                bsrSortedRowPtrA,
                                                 const int*                bsrSortedColIndA,
                                                 int                       blockDim,
                                                 bsrsv2Info_t              info,
                                                 size_t*                   pBufferSize)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_zbsrsv_buffer_size((rocsparse_handle)handle,
                                     hipDirectionToHCCDirection(dir),
                                     hipOperationToHCCOperation(transA),
                                     mb,
                                     nnzb,
                                     (rocsparse_mat_descr)descrA,
                                     (rocsparse_double_complex*)bsrSortedValA,
                                     bsrSortedRowPtrA,
                                     bsrSortedColIndA,
                                     blockDim,
                                     (rocsparse_mat_info)info,
                                     pBufferSize));
}

hipsparseStatus_t hipsparseSbsrsv2_analysis(hipsparseHandle_t         handle,
                                            hipsparseDirection_t      dir,
                                            hipsparseOperation_t      transA,
                                            int                       mb,
                                            int                       nnzb,
                                            const hipsparseMatDescr_t descrA,
                                            const float*              bsrSortedValA,
                                            const int*                bsrSortedRowPtrA,
                                            const int*                bsrSortedColIndA,
                                            int                       blockDim,
                                            bsrsv2Info_t              info,
                                            hipsparseSolvePolicy_t    policy,
                                            void*                     pBuffer)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_sbsrsv_analysis((rocsparse_handle)handle,
                                                                hipDirectionToHCCDirection(dir),
                                                                hipOperationToHCCOperation(transA),
                                                                mb,
                                                                nnzb,
                                                                (rocsparse_mat_descr)descrA,
                                                                bsrSortedValA,
                                                                bsrSortedRowPtrA,
                                                                bsrSortedColIndA,
                                                                blockDim,
                                                                (rocsparse_mat_info)info,
                                                                rocsparse_analysis_policy_force,
                                                                rocsparse_solve_policy_auto,
                                                                pBuffer));
}

hipsparseStatus_t hipsparseDbsrsv2_analysis(hipsparseHandle_t         handle,
                                            hipsparseDirection_t      dir,
                                            hipsparseOperation_t      transA,
                                            int                       mb,
                                            int                       nnzb,
                                            const hipsparseMatDescr_t descrA,
                                            const double*             bsrSortedValA,
                                            const int*                bsrSortedRowPtrA,
                                            const int*                bsrSortedColIndA,
                                            int                       blockDim,
                                            bsrsv2Info_t              info,
                                            hipsparseSolvePolicy_t    policy,
                                            void*                     pBuffer)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_dbsrsv_analysis((rocsparse_handle)handle,
                                                                hipDirectionToHCCDirection(dir),
                                                                hipOperationToHCCOperation(transA),
                                                                mb,
                                                                nnzb,
                                                                (rocsparse_mat_descr)descrA,
                                                                bsrSortedValA,
                                                                bsrSortedRowPtrA,
                                                                bsrSortedColIndA,
                                                                blockDim,
                                                                (rocsparse_mat_info)info,
                                                                rocsparse_analysis_policy_force,
                                                                rocsparse_solve_policy_auto,
                                                                pBuffer));
}

hipsparseStatus_t hipsparseCbsrsv2_analysis(hipsparseHandle_t         handle,
                                            hipsparseDirection_t      dir,
                                            hipsparseOperation_t      transA,
                                            int                       mb,
                                            int                       nnzb,
                                            const hipsparseMatDescr_t descrA,
                                            const hipComplex*         bsrSortedValA,
                                            const int*                bsrSortedRowPtrA,
                                            const int*                bsrSortedColIndA,
                                            int                       blockDim,
                                            bsrsv2Info_t              info,
                                            hipsparseSolvePolicy_t    policy,
                                            void*                     pBuffer)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_cbsrsv_analysis((rocsparse_handle)handle,
                                  hipDirectionToHCCDirection(dir),
                                  hipOperationToHCCOperation(transA),
                                  mb,
                                  nnzb,
                                  (rocsparse_mat_descr)descrA,
                                  (const rocsparse_float_complex*)bsrSortedValA,
                                  bsrSortedRowPtrA,
                                  bsrSortedColIndA,
                                  blockDim,
                                  (rocsparse_mat_info)info,
                                  rocsparse_analysis_policy_force,
                                  rocsparse_solve_policy_auto,
                                  pBuffer));
}

hipsparseStatus_t hipsparseZbsrsv2_analysis(hipsparseHandle_t         handle,
                                            hipsparseDirection_t      dir,
                                            hipsparseOperation_t      transA,
                                            int                       mb,
                                            int                       nnzb,
                                            const hipsparseMatDescr_t descrA,
                                            const hipDoubleComplex*   bsrSortedValA,
                                            const int*                bsrSortedRowPtrA,
                                            const int*                bsrSortedColIndA,
                                            int                       blockDim,
                                            bsrsv2Info_t              info,
                                            hipsparseSolvePolicy_t    policy,
                                            void*                     pBuffer)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_zbsrsv_analysis((rocsparse_handle)handle,
                                  hipDirectionToHCCDirection(dir),
                                  hipOperationToHCCOperation(transA),
                                  mb,
                                  nnzb,
                                  (rocsparse_mat_descr)descrA,
                                  (const rocsparse_double_complex*)bsrSortedValA,
                                  bsrSortedRowPtrA,
                                  bsrSortedColIndA,
                                  blockDim,
                                  (rocsparse_mat_info)info,
                                  rocsparse_analysis_policy_force,
                                  rocsparse_solve_policy_auto,
                                  pBuffer));
}

hipsparseStatus_t hipsparseSbsrsv2_solve(hipsparseHandle_t         handle,
                                         hipsparseDirection_t      dir,
                                         hipsparseOperation_t      transA,
                                         int                       mb,
                                         int                       nnzb,
                                         const float*              alpha,
                                         const hipsparseMatDescr_t descrA,
                                         const float*              bsrSortedValA,
                                         const int*                bsrSortedRowPtrA,
                                         const int*                bsrSortedColIndA,
                                         int                       blockDim,
                                         bsrsv2Info_t              info,
                                         const float*              f,
                                         float*                    x,
                                         hipsparseSolvePolicy_t    policy,
                                         void*                     pBuffer)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_sbsrsv_solve((rocsparse_handle)handle,
                                                             hipDirectionToHCCDirection(dir),
                                                             hipOperationToHCCOperation(transA),
                                                             mb,
                                                             nnzb,
                                                             alpha,
                                                             (rocsparse_mat_descr)descrA,
                                                             bsrSortedValA,
                                                             bsrSortedRowPtrA,
                                                             bsrSortedColIndA,
                                                             blockDim,
                                                             (rocsparse_mat_info)info,
                                                             f,
                                                             x,
                                                             rocsparse_solve_policy_auto,
                                                             pBuffer));
}

hipsparseStatus_t hipsparseDbsrsv2_solve(hipsparseHandle_t         handle,
                                         hipsparseDirection_t      dir,
                                         hipsparseOperation_t      transA,
                                         int                       mb,
                                         int                       nnzb,
                                         const double*             alpha,
                                         const hipsparseMatDescr_t descrA,
                                         const double*             bsrSortedValA,
                                         const int*                bsrSortedRowPtrA,
                                         const int*                bsrSortedColIndA,
                                         int                       blockDim,
                                         bsrsv2Info_t              info,
                                         const double*             f,
                                         double*                   x,
                                         hipsparseSolvePolicy_t    policy,
                                         void*                     pBuffer)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_dbsrsv_solve((rocsparse_handle)handle,
                                                             hipDirectionToHCCDirection(dir),
                                                             hipOperationToHCCOperation(transA),
                                                             mb,
                                                             nnzb,
                                                             alpha,
                                                             (rocsparse_mat_descr)descrA,
                                                             bsrSortedValA,
                                                             bsrSortedRowPtrA,
                                                             bsrSortedColIndA,
                                                             blockDim,
                                                             (rocsparse_mat_info)info,
                                                             f,
                                                             x,
                                                             rocsparse_solve_policy_auto,
                                                             pBuffer));
}

hipsparseStatus_t hipsparseCbsrsv2_solve(hipsparseHandle_t         handle,
                                         hipsparseDirection_t      dir,
                                         hipsparseOperation_t      transA,
                                         int                       mb,
                                         int                       nnzb,
                                         const hipComplex*         alpha,
                                         const hipsparseMatDescr_t descrA,
                                         const hipComplex*         bsrSortedValA,
                                         const int*                bsrSortedRowPtrA,
                                         const int*                bsrSortedColIndA,
                                         int                       blockDim,
                                         bsrsv2Info_t              info,
                                         const hipComplex*         f,
                                         hipComplex*               x,
                                         hipsparseSolvePolicy_t    policy,
                                         void*                     pBuffer)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_cbsrsv_solve((rocsparse_handle)handle,
                               hipDirectionToHCCDirection(dir),
                               hipOperationToHCCOperation(transA),
                               mb,
                               nnzb,
                               (const rocsparse_float_complex*)alpha,
                               (rocsparse_mat_descr)descrA,
                               (const rocsparse_float_complex*)bsrSortedValA,
                               bsrSortedRowPtrA,
                               bsrSortedColIndA,
                               blockDim,
                               (rocsparse_mat_info)info,
                               (const rocsparse_float_complex*)f,
                               (rocsparse_float_complex*)x,
                               rocsparse_solve_policy_auto,
                               pBuffer));
}

hipsparseStatus_t hipsparseZbsrsv2_solve(hipsparseHandle_t         handle,
                                         hipsparseDirection_t      dir,
                                         hipsparseOperation_t      transA,
                                         int                       mb,
                                         int                       nnzb,
                                         const hipDoubleComplex*   alpha,
                                         const hipsparseMatDescr_t descrA,
                                         const hipDoubleComplex*   bsrSortedValA,
                                         const int*                bsrSortedRowPtrA,
                                         const int*                bsrSortedColIndA,
                                         int                       blockDim,
                                         bsrsv2Info_t              info,
                                         const hipDoubleComplex*   f,
                                         hipDoubleComplex*         x,
                                         hipsparseSolvePolicy_t    policy,
                                         void*                     pBuffer)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_zbsrsv_solve((rocsparse_handle)handle,
                               hipDirectionToHCCDirection(dir),
                               hipOperationToHCCOperation(transA),
                               mb,
                               nnzb,
                               (const rocsparse_double_complex*)alpha,
                               (rocsparse_mat_descr)descrA,
                               (const rocsparse_double_complex*)bsrSortedValA,
                               bsrSortedRowPtrA,
                               bsrSortedColIndA,
                               blockDim,
                               (rocsparse_mat_info)info,
                               (const rocsparse_double_complex*)f,
                               (rocsparse_double_complex*)x,
                               rocsparse_solve_policy_auto,
                               pBuffer));
}

hipsparseStatus_t hipsparseSgemvi_bufferSize(
    hipsparseHandle_t handle, hipsparseOperation_t transA, int m, int n, int nnz, int* pBufferSize)
{
    if(pBufferSize == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    size_t buffer_size;

    hipsparseStatus_t status = rocSPARSEStatusToHIPStatus(rocsparse_sgemvi_buffer_size(
        (rocsparse_handle)handle, hipOperationToHCCOperation(transA), m, n, nnz, &buffer_size));

    *pBufferSize = (int)buffer_size;

    return status;
}

hipsparseStatus_t hipsparseDgemvi_bufferSize(
    hipsparseHandle_t handle, hipsparseOperation_t transA, int m, int n, int nnz, int* pBufferSize)
{
    if(pBufferSize == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    size_t buffer_size;

    hipsparseStatus_t status = rocSPARSEStatusToHIPStatus(rocsparse_dgemvi_buffer_size(
        (rocsparse_handle)handle, hipOperationToHCCOperation(transA), m, n, nnz, &buffer_size));

    *pBufferSize = (int)buffer_size;

    return status;
}

hipsparseStatus_t hipsparseCgemvi_bufferSize(
    hipsparseHandle_t handle, hipsparseOperation_t transA, int m, int n, int nnz, int* pBufferSize)
{
    if(pBufferSize == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    size_t buffer_size;

    hipsparseStatus_t status = rocSPARSEStatusToHIPStatus(rocsparse_cgemvi_buffer_size(
        (rocsparse_handle)handle, hipOperationToHCCOperation(transA), m, n, nnz, &buffer_size));

    *pBufferSize = (int)buffer_size;

    return status;
}

hipsparseStatus_t hipsparseZgemvi_bufferSize(
    hipsparseHandle_t handle, hipsparseOperation_t transA, int m, int n, int nnz, int* pBufferSize)
{
    if(pBufferSize == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    size_t buffer_size;

    hipsparseStatus_t status = rocSPARSEStatusToHIPStatus(rocsparse_zgemvi_buffer_size(
        (rocsparse_handle)handle, hipOperationToHCCOperation(transA), m, n, nnz, &buffer_size));

    *pBufferSize = (int)buffer_size;

    return status;
}

hipsparseStatus_t hipsparseSgemvi(hipsparseHandle_t    handle,
                                  hipsparseOperation_t transA,
                                  int                  m,
                                  int                  n,
                                  const float*         alpha,
                                  const float*         A,
                                  int                  lda,
                                  int                  nnz,
                                  const float*         x,
                                  const int*           xInd,
                                  const float*         beta,
                                  float*               y,
                                  hipsparseIndexBase_t idxBase,
                                  void*                pBuffer)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_sgemvi((rocsparse_handle)handle,
                                                       hipOperationToHCCOperation(transA),
                                                       m,
                                                       n,
                                                       alpha,
                                                       A,
                                                       lda,
                                                       nnz,
                                                       x,
                                                       xInd,
                                                       beta,
                                                       y,
                                                       hipBaseToHCCBase(idxBase),
                                                       pBuffer));
}

hipsparseStatus_t hipsparseDgemvi(hipsparseHandle_t    handle,
                                  hipsparseOperation_t transA,
                                  int                  m,
                                  int                  n,
                                  const double*        alpha,
                                  const double*        A,
                                  int                  lda,
                                  int                  nnz,
                                  const double*        x,
                                  const int*           xInd,
                                  const double*        beta,
                                  double*              y,
                                  hipsparseIndexBase_t idxBase,
                                  void*                pBuffer)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_dgemvi((rocsparse_handle)handle,
                                                       hipOperationToHCCOperation(transA),
                                                       m,
                                                       n,
                                                       alpha,
                                                       A,
                                                       lda,
                                                       nnz,
                                                       x,
                                                       xInd,
                                                       beta,
                                                       y,
                                                       hipBaseToHCCBase(idxBase),
                                                       pBuffer));
}

hipsparseStatus_t hipsparseCgemvi(hipsparseHandle_t    handle,
                                  hipsparseOperation_t transA,
                                  int                  m,
                                  int                  n,
                                  const hipComplex*    alpha,
                                  const hipComplex*    A,
                                  int                  lda,
                                  int                  nnz,
                                  const hipComplex*    x,
                                  const int*           xInd,
                                  const hipComplex*    beta,
                                  hipComplex*          y,
                                  hipsparseIndexBase_t idxBase,
                                  void*                pBuffer)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_cgemvi((rocsparse_handle)handle,
                                                       hipOperationToHCCOperation(transA),
                                                       m,
                                                       n,
                                                       (const rocsparse_float_complex*)alpha,
                                                       (const rocsparse_float_complex*)A,
                                                       lda,
                                                       nnz,
                                                       (const rocsparse_float_complex*)x,
                                                       xInd,
                                                       (const rocsparse_float_complex*)beta,
                                                       (rocsparse_float_complex*)y,
                                                       hipBaseToHCCBase(idxBase),
                                                       pBuffer));
}

hipsparseStatus_t hipsparseZgemvi(hipsparseHandle_t       handle,
                                  hipsparseOperation_t    transA,
                                  int                     m,
                                  int                     n,
                                  const hipDoubleComplex* alpha,
                                  const hipDoubleComplex* A,
                                  int                     lda,
                                  int                     nnz,
                                  const hipDoubleComplex* x,
                                  const int*              xInd,
                                  const hipDoubleComplex* beta,
                                  hipDoubleComplex*       y,
                                  hipsparseIndexBase_t    idxBase,
                                  void*                   pBuffer)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_zgemvi((rocsparse_handle)handle,
                                                       hipOperationToHCCOperation(transA),
                                                       m,
                                                       n,
                                                       (const rocsparse_double_complex*)alpha,
                                                       (const rocsparse_double_complex*)A,
                                                       lda,
                                                       nnz,
                                                       (const rocsparse_double_complex*)x,
                                                       xInd,
                                                       (const rocsparse_double_complex*)beta,
                                                       (rocsparse_double_complex*)y,
                                                       hipBaseToHCCBase(idxBase),
                                                       pBuffer));
}

hipsparseStatus_t hipsparseSbsrmm(hipsparseHandle_t         handle,
                                  hipsparseDirection_t      dirA,
                                  hipsparseOperation_t      transA,
                                  hipsparseOperation_t      transB,
                                  int                       mb,
                                  int                       n,
                                  int                       kb,
                                  int                       nnzb,
                                  const float*              alpha,
                                  const hipsparseMatDescr_t descrA,
                                  const float*              bsrValA,
                                  const int*                bsrRowPtrA,
                                  const int*                bsrColIndA,
                                  int                       blockDim,
                                  const float*              B,
                                  int                       ldb,
                                  const float*              beta,
                                  float*                    C,
                                  int                       ldc)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_sbsrmm((rocsparse_handle)handle,
                                                       hipDirectionToHCCDirection(dirA),
                                                       hipOperationToHCCOperation(transA),
                                                       hipOperationToHCCOperation(transB),
                                                       mb,
                                                       n,
                                                       kb,
                                                       nnzb,
                                                       alpha,
                                                       (rocsparse_mat_descr)descrA,
                                                       bsrValA,
                                                       bsrRowPtrA,
                                                       bsrColIndA,
                                                       blockDim,
                                                       B,
                                                       ldb,
                                                       beta,
                                                       C,
                                                       ldc));
}

hipsparseStatus_t hipsparseDbsrmm(hipsparseHandle_t         handle,
                                  hipsparseDirection_t      dirA,
                                  hipsparseOperation_t      transA,
                                  hipsparseOperation_t      transB,
                                  int                       mb,
                                  int                       n,
                                  int                       kb,
                                  int                       nnzb,
                                  const double*             alpha,
                                  const hipsparseMatDescr_t descrA,
                                  const double*             bsrValA,
                                  const int*                bsrRowPtrA,
                                  const int*                bsrColIndA,
                                  int                       blockDim,
                                  const double*             B,
                                  int                       ldb,
                                  const double*             beta,
                                  double*                   C,
                                  int                       ldc)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_dbsrmm((rocsparse_handle)handle,
                                                       hipDirectionToHCCDirection(dirA),
                                                       hipOperationToHCCOperation(transA),
                                                       hipOperationToHCCOperation(transB),
                                                       mb,
                                                       n,
                                                       kb,
                                                       nnzb,
                                                       alpha,
                                                       (rocsparse_mat_descr)descrA,
                                                       bsrValA,
                                                       bsrRowPtrA,
                                                       bsrColIndA,
                                                       blockDim,
                                                       B,
                                                       ldb,
                                                       beta,
                                                       C,
                                                       ldc));
}

hipsparseStatus_t hipsparseCbsrmm(hipsparseHandle_t         handle,
                                  hipsparseDirection_t      dirA,
                                  hipsparseOperation_t      transA,
                                  hipsparseOperation_t      transB,
                                  int                       mb,
                                  int                       n,
                                  int                       kb,
                                  int                       nnzb,
                                  const hipComplex*         alpha,
                                  const hipsparseMatDescr_t descrA,
                                  const hipComplex*         bsrValA,
                                  const int*                bsrRowPtrA,
                                  const int*                bsrColIndA,
                                  int                       blockDim,
                                  const hipComplex*         B,
                                  int                       ldb,
                                  const hipComplex*         beta,
                                  hipComplex*               C,
                                  int                       ldc)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_cbsrmm((rocsparse_handle)handle,
                                                       hipDirectionToHCCDirection(dirA),
                                                       hipOperationToHCCOperation(transA),
                                                       hipOperationToHCCOperation(transB),
                                                       mb,
                                                       n,
                                                       kb,
                                                       nnzb,
                                                       (const rocsparse_float_complex*)alpha,
                                                       (rocsparse_mat_descr)descrA,
                                                       (const rocsparse_float_complex*)bsrValA,
                                                       bsrRowPtrA,
                                                       bsrColIndA,
                                                       blockDim,
                                                       (const rocsparse_float_complex*)B,
                                                       ldb,
                                                       (const rocsparse_float_complex*)beta,
                                                       (rocsparse_float_complex*)C,
                                                       ldc));
}

hipsparseStatus_t hipsparseZbsrmm(hipsparseHandle_t         handle,
                                  hipsparseDirection_t      dirA,
                                  hipsparseOperation_t      transA,
                                  hipsparseOperation_t      transB,
                                  int                       mb,
                                  int                       n,
                                  int                       kb,
                                  int                       nnzb,
                                  const hipDoubleComplex*   alpha,
                                  const hipsparseMatDescr_t descrA,
                                  const hipDoubleComplex*   bsrValA,
                                  const int*                bsrRowPtrA,
                                  const int*                bsrColIndA,
                                  int                       blockDim,
                                  const hipDoubleComplex*   B,
                                  int                       ldb,
                                  const hipDoubleComplex*   beta,
                                  hipDoubleComplex*         C,
                                  int                       ldc)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_zbsrmm((rocsparse_handle)handle,
                                                       hipDirectionToHCCDirection(dirA),
                                                       hipOperationToHCCOperation(transA),
                                                       hipOperationToHCCOperation(transB),
                                                       mb,
                                                       n,
                                                       kb,
                                                       nnzb,
                                                       (const rocsparse_double_complex*)alpha,
                                                       (rocsparse_mat_descr)descrA,
                                                       (const rocsparse_double_complex*)bsrValA,
                                                       bsrRowPtrA,
                                                       bsrColIndA,
                                                       blockDim,
                                                       (const rocsparse_double_complex*)B,
                                                       ldb,
                                                       (const rocsparse_double_complex*)beta,
                                                       (rocsparse_double_complex*)C,
                                                       ldc));
}

hipsparseStatus_t hipsparseScsrmm(hipsparseHandle_t         handle,
                                  hipsparseOperation_t      transA,
                                  int                       m,
                                  int                       n,
                                  int                       k,
                                  int                       nnz,
                                  const float*              alpha,
                                  const hipsparseMatDescr_t descrA,
                                  const float*              csrSortedValA,
                                  const int*                csrSortedRowPtrA,
                                  const int*                csrSortedColIndA,
                                  const float*              B,
                                  int                       ldb,
                                  const float*              beta,
                                  float*                    C,
                                  int                       ldc)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_scsrmm((rocsparse_handle)handle,
                                                       hipOperationToHCCOperation(transA),
                                                       rocsparse_operation_none,
                                                       m,
                                                       n,
                                                       k,
                                                       nnz,
                                                       alpha,
                                                       (rocsparse_mat_descr)descrA,
                                                       csrSortedValA,
                                                       csrSortedRowPtrA,
                                                       csrSortedColIndA,
                                                       B,
                                                       ldb,
                                                       beta,
                                                       C,
                                                       ldc));
}

hipsparseStatus_t hipsparseDcsrmm(hipsparseHandle_t         handle,
                                  hipsparseOperation_t      transA,
                                  int                       m,
                                  int                       n,
                                  int                       k,
                                  int                       nnz,
                                  const double*             alpha,
                                  const hipsparseMatDescr_t descrA,
                                  const double*             csrSortedValA,
                                  const int*                csrSortedRowPtrA,
                                  const int*                csrSortedColIndA,
                                  const double*             B,
                                  int                       ldb,
                                  const double*             beta,
                                  double*                   C,
                                  int                       ldc)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_dcsrmm((rocsparse_handle)handle,
                                                       hipOperationToHCCOperation(transA),
                                                       rocsparse_operation_none,
                                                       m,
                                                       n,
                                                       k,
                                                       nnz,
                                                       alpha,
                                                       (rocsparse_mat_descr)descrA,
                                                       csrSortedValA,
                                                       csrSortedRowPtrA,
                                                       csrSortedColIndA,
                                                       B,
                                                       ldb,
                                                       beta,
                                                       C,
                                                       ldc));
}

hipsparseStatus_t hipsparseCcsrmm(hipsparseHandle_t         handle,
                                  hipsparseOperation_t      transA,
                                  int                       m,
                                  int                       n,
                                  int                       k,
                                  int                       nnz,
                                  const hipComplex*         alpha,
                                  const hipsparseMatDescr_t descrA,
                                  const hipComplex*         csrSortedValA,
                                  const int*                csrSortedRowPtrA,
                                  const int*                csrSortedColIndA,
                                  const hipComplex*         B,
                                  int                       ldb,
                                  const hipComplex*         beta,
                                  hipComplex*               C,
                                  int                       ldc)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_ccsrmm((rocsparse_handle)handle,
                         hipOperationToHCCOperation(transA),
                         rocsparse_operation_none,
                         m,
                         n,
                         k,
                         nnz,
                         (const rocsparse_float_complex*)alpha,
                         (rocsparse_mat_descr)descrA,
                         (const rocsparse_float_complex*)csrSortedValA,
                         csrSortedRowPtrA,
                         csrSortedColIndA,
                         (const rocsparse_float_complex*)B,
                         ldb,
                         (const rocsparse_float_complex*)beta,
                         (rocsparse_float_complex*)C,
                         ldc));
}

hipsparseStatus_t hipsparseZcsrmm(hipsparseHandle_t         handle,
                                  hipsparseOperation_t      transA,
                                  int                       m,
                                  int                       n,
                                  int                       k,
                                  int                       nnz,
                                  const hipDoubleComplex*   alpha,
                                  const hipsparseMatDescr_t descrA,
                                  const hipDoubleComplex*   csrSortedValA,
                                  const int*                csrSortedRowPtrA,
                                  const int*                csrSortedColIndA,
                                  const hipDoubleComplex*   B,
                                  int                       ldb,
                                  const hipDoubleComplex*   beta,
                                  hipDoubleComplex*         C,
                                  int                       ldc)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_zcsrmm((rocsparse_handle)handle,
                         hipOperationToHCCOperation(transA),
                         rocsparse_operation_none,
                         m,
                         n,
                         k,
                         nnz,
                         (const rocsparse_double_complex*)alpha,
                         (rocsparse_mat_descr)descrA,
                         (const rocsparse_double_complex*)csrSortedValA,
                         csrSortedRowPtrA,
                         csrSortedColIndA,
                         (const rocsparse_double_complex*)B,
                         ldb,
                         (const rocsparse_double_complex*)beta,
                         (rocsparse_double_complex*)C,
                         ldc));
}

hipsparseStatus_t hipsparseScsrmm2(hipsparseHandle_t         handle,
                                   hipsparseOperation_t      transA,
                                   hipsparseOperation_t      transB,
                                   int                       m,
                                   int                       n,
                                   int                       k,
                                   int                       nnz,
                                   const float*              alpha,
                                   const hipsparseMatDescr_t descrA,
                                   const float*              csrSortedValA,
                                   const int*                csrSortedRowPtrA,
                                   const int*                csrSortedColIndA,
                                   const float*              B,
                                   int                       ldb,
                                   const float*              beta,
                                   float*                    C,
                                   int                       ldc)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_scsrmm((rocsparse_handle)handle,
                                                       hipOperationToHCCOperation(transA),
                                                       hipOperationToHCCOperation(transB),
                                                       m,
                                                       n,
                                                       k,
                                                       nnz,
                                                       alpha,
                                                       (rocsparse_mat_descr)descrA,
                                                       csrSortedValA,
                                                       csrSortedRowPtrA,
                                                       csrSortedColIndA,
                                                       B,
                                                       ldb,
                                                       beta,
                                                       C,
                                                       ldc));
}

hipsparseStatus_t hipsparseDcsrmm2(hipsparseHandle_t         handle,
                                   hipsparseOperation_t      transA,
                                   hipsparseOperation_t      transB,
                                   int                       m,
                                   int                       n,
                                   int                       k,
                                   int                       nnz,
                                   const double*             alpha,
                                   const hipsparseMatDescr_t descrA,
                                   const double*             csrSortedValA,
                                   const int*                csrSortedRowPtrA,
                                   const int*                csrSortedColIndA,
                                   const double*             B,
                                   int                       ldb,
                                   const double*             beta,
                                   double*                   C,
                                   int                       ldc)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_dcsrmm((rocsparse_handle)handle,
                                                       hipOperationToHCCOperation(transA),
                                                       hipOperationToHCCOperation(transB),
                                                       m,
                                                       n,
                                                       k,
                                                       nnz,
                                                       alpha,
                                                       (rocsparse_mat_descr)descrA,
                                                       csrSortedValA,
                                                       csrSortedRowPtrA,
                                                       csrSortedColIndA,
                                                       B,
                                                       ldb,
                                                       beta,
                                                       C,
                                                       ldc));
}

hipsparseStatus_t hipsparseCcsrmm2(hipsparseHandle_t         handle,
                                   hipsparseOperation_t      transA,
                                   hipsparseOperation_t      transB,
                                   int                       m,
                                   int                       n,
                                   int                       k,
                                   int                       nnz,
                                   const hipComplex*         alpha,
                                   const hipsparseMatDescr_t descrA,
                                   const hipComplex*         csrSortedValA,
                                   const int*                csrSortedRowPtrA,
                                   const int*                csrSortedColIndA,
                                   const hipComplex*         B,
                                   int                       ldb,
                                   const hipComplex*         beta,
                                   hipComplex*               C,
                                   int                       ldc)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_ccsrmm((rocsparse_handle)handle,
                         hipOperationToHCCOperation(transA),
                         hipOperationToHCCOperation(transB),
                         m,
                         n,
                         k,
                         nnz,
                         (const rocsparse_float_complex*)alpha,
                         (rocsparse_mat_descr)descrA,
                         (const rocsparse_float_complex*)csrSortedValA,
                         csrSortedRowPtrA,
                         csrSortedColIndA,
                         (const rocsparse_float_complex*)B,
                         ldb,
                         (const rocsparse_float_complex*)beta,
                         (rocsparse_float_complex*)C,
                         ldc));
}

hipsparseStatus_t hipsparseZcsrmm2(hipsparseHandle_t         handle,
                                   hipsparseOperation_t      transA,
                                   hipsparseOperation_t      transB,
                                   int                       m,
                                   int                       n,
                                   int                       k,
                                   int                       nnz,
                                   const hipDoubleComplex*   alpha,
                                   const hipsparseMatDescr_t descrA,
                                   const hipDoubleComplex*   csrSortedValA,
                                   const int*                csrSortedRowPtrA,
                                   const int*                csrSortedColIndA,
                                   const hipDoubleComplex*   B,
                                   int                       ldb,
                                   const hipDoubleComplex*   beta,
                                   hipDoubleComplex*         C,
                                   int                       ldc)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_zcsrmm((rocsparse_handle)handle,
                         hipOperationToHCCOperation(transA),
                         hipOperationToHCCOperation(transB),
                         m,
                         n,
                         k,
                         nnz,
                         (const rocsparse_double_complex*)alpha,
                         (rocsparse_mat_descr)descrA,
                         (const rocsparse_double_complex*)csrSortedValA,
                         csrSortedRowPtrA,
                         csrSortedColIndA,
                         (const rocsparse_double_complex*)B,
                         ldb,
                         (const rocsparse_double_complex*)beta,
                         (rocsparse_double_complex*)C,
                         ldc));
}

hipsparseStatus_t
    hipsparseXbsrsm2_zeroPivot(hipsparseHandle_t handle, bsrsm2Info_t info, int* position)
{
    // Obtain stream, to explicitly sync (cusparse bsrsm2_zeropivot is blocking)
    hipStream_t stream;
    RETURN_IF_HIPSPARSE_ERROR(hipsparseGetStream(handle, &stream));

    // bsrsm zero pivot
    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse_bsrsm_zero_pivot((rocsparse_handle)handle, (rocsparse_mat_info)info, position));

    // Synchronize stream
    RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseSbsrsm2_bufferSize(hipsparseHandle_t         handle,
                                              hipsparseDirection_t      dirA,
                                              hipsparseOperation_t      transA,
                                              hipsparseOperation_t      transX,
                                              int                       mb,
                                              int                       nrhs,
                                              int                       nnzb,
                                              const hipsparseMatDescr_t descrA,
                                              float*                    bsrSortedValA,
                                              const int*                bsrSortedRowPtrA,
                                              const int*                bsrSortedColIndA,
                                              int                       blockDim,
                                              bsrsm2Info_t              info,
                                              int*                      pBufferSizeInBytes)
{
    if(pBufferSizeInBytes == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    size_t size;

    hipsparseStatus_t status = rocSPARSEStatusToHIPStatus(
        rocsparse_sbsrsm_buffer_size((rocsparse_handle)handle,
                                     hipDirectionToHCCDirection(dirA),
                                     hipOperationToHCCOperation(transA),
                                     hipOperationToHCCOperation(transX),
                                     mb,
                                     nrhs,
                                     nnzb,
                                     (const rocsparse_mat_descr)descrA,
                                     bsrSortedValA,
                                     bsrSortedRowPtrA,
                                     bsrSortedColIndA,
                                     blockDim,
                                     (rocsparse_mat_info)info,
                                     &size));

    *pBufferSizeInBytes = (int)size;

    return status;
}

hipsparseStatus_t hipsparseDbsrsm2_bufferSize(hipsparseHandle_t         handle,
                                              hipsparseDirection_t      dirA,
                                              hipsparseOperation_t      transA,
                                              hipsparseOperation_t      transX,
                                              int                       mb,
                                              int                       nrhs,
                                              int                       nnzb,
                                              const hipsparseMatDescr_t descrA,
                                              double*                   bsrSortedValA,
                                              const int*                bsrSortedRowPtrA,
                                              const int*                bsrSortedColIndA,
                                              int                       blockDim,
                                              bsrsm2Info_t              info,
                                              int*                      pBufferSizeInBytes)
{
    if(pBufferSizeInBytes == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    size_t size;

    hipsparseStatus_t status = rocSPARSEStatusToHIPStatus(
        rocsparse_dbsrsm_buffer_size((rocsparse_handle)handle,
                                     hipDirectionToHCCDirection(dirA),
                                     hipOperationToHCCOperation(transA),
                                     hipOperationToHCCOperation(transX),
                                     mb,
                                     nrhs,
                                     nnzb,
                                     (const rocsparse_mat_descr)descrA,
                                     bsrSortedValA,
                                     bsrSortedRowPtrA,
                                     bsrSortedColIndA,
                                     blockDim,
                                     (rocsparse_mat_info)info,
                                     &size));

    *pBufferSizeInBytes = (int)size;

    return status;
}

hipsparseStatus_t hipsparseCbsrsm2_bufferSize(hipsparseHandle_t         handle,
                                              hipsparseDirection_t      dirA,
                                              hipsparseOperation_t      transA,
                                              hipsparseOperation_t      transX,
                                              int                       mb,
                                              int                       nrhs,
                                              int                       nnzb,
                                              const hipsparseMatDescr_t descrA,
                                              hipComplex*               bsrSortedValA,
                                              const int*                bsrSortedRowPtrA,
                                              const int*                bsrSortedColIndA,
                                              int                       blockDim,
                                              bsrsm2Info_t              info,
                                              int*                      pBufferSizeInBytes)
{
    if(pBufferSizeInBytes == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    size_t size;

    hipsparseStatus_t status = rocSPARSEStatusToHIPStatus(
        rocsparse_cbsrsm_buffer_size((rocsparse_handle)handle,
                                     hipDirectionToHCCDirection(dirA),
                                     hipOperationToHCCOperation(transA),
                                     hipOperationToHCCOperation(transX),
                                     mb,
                                     nrhs,
                                     nnzb,
                                     (const rocsparse_mat_descr)descrA,
                                     (rocsparse_float_complex*)bsrSortedValA,
                                     bsrSortedRowPtrA,
                                     bsrSortedColIndA,
                                     blockDim,
                                     (rocsparse_mat_info)info,
                                     &size));

    *pBufferSizeInBytes = (int)size;

    return status;
}

hipsparseStatus_t hipsparseZbsrsm2_bufferSize(hipsparseHandle_t         handle,
                                              hipsparseDirection_t      dirA,
                                              hipsparseOperation_t      transA,
                                              hipsparseOperation_t      transX,
                                              int                       mb,
                                              int                       nrhs,
                                              int                       nnzb,
                                              const hipsparseMatDescr_t descrA,
                                              hipDoubleComplex*         bsrSortedValA,
                                              const int*                bsrSortedRowPtrA,
                                              const int*                bsrSortedColIndA,
                                              int                       blockDim,
                                              bsrsm2Info_t              info,
                                              int*                      pBufferSizeInBytes)
{
    if(pBufferSizeInBytes == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    size_t size;

    hipsparseStatus_t status = rocSPARSEStatusToHIPStatus(
        rocsparse_zbsrsm_buffer_size((rocsparse_handle)handle,
                                     hipDirectionToHCCDirection(dirA),
                                     hipOperationToHCCOperation(transA),
                                     hipOperationToHCCOperation(transX),
                                     mb,
                                     nrhs,
                                     nnzb,
                                     (const rocsparse_mat_descr)descrA,
                                     (rocsparse_double_complex*)bsrSortedValA,
                                     bsrSortedRowPtrA,
                                     bsrSortedColIndA,
                                     blockDim,
                                     (rocsparse_mat_info)info,
                                     &size));

    *pBufferSizeInBytes = (int)size;

    return status;
}

hipsparseStatus_t hipsparseSbsrsm2_analysis(hipsparseHandle_t         handle,
                                            hipsparseDirection_t      dirA,
                                            hipsparseOperation_t      transA,
                                            hipsparseOperation_t      transX,
                                            int                       mb,
                                            int                       nrhs,
                                            int                       nnzb,
                                            const hipsparseMatDescr_t descrA,
                                            const float*              bsrSortedValA,
                                            const int*                bsrSortedRowPtrA,
                                            const int*                bsrSortedColIndA,
                                            int                       blockDim,
                                            bsrsm2Info_t              info,
                                            hipsparseSolvePolicy_t    policy,
                                            void*                     pBuffer)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_sbsrsm_analysis((rocsparse_handle)handle,
                                                                hipDirectionToHCCDirection(dirA),
                                                                hipOperationToHCCOperation(transA),
                                                                hipOperationToHCCOperation(transX),
                                                                mb,
                                                                nrhs,
                                                                nnzb,
                                                                (const rocsparse_mat_descr)descrA,
                                                                bsrSortedValA,
                                                                bsrSortedRowPtrA,
                                                                bsrSortedColIndA,
                                                                blockDim,
                                                                (rocsparse_mat_info)info,
                                                                rocsparse_analysis_policy_force,
                                                                rocsparse_solve_policy_auto,
                                                                pBuffer));
}

hipsparseStatus_t hipsparseDbsrsm2_analysis(hipsparseHandle_t         handle,
                                            hipsparseDirection_t      dirA,
                                            hipsparseOperation_t      transA,
                                            hipsparseOperation_t      transX,
                                            int                       mb,
                                            int                       nrhs,
                                            int                       nnzb,
                                            const hipsparseMatDescr_t descrA,
                                            const double*             bsrSortedValA,
                                            const int*                bsrSortedRowPtrA,
                                            const int*                bsrSortedColIndA,
                                            int                       blockDim,
                                            bsrsm2Info_t              info,
                                            hipsparseSolvePolicy_t    policy,
                                            void*                     pBuffer)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_dbsrsm_analysis((rocsparse_handle)handle,
                                                                hipDirectionToHCCDirection(dirA),
                                                                hipOperationToHCCOperation(transA),
                                                                hipOperationToHCCOperation(transX),
                                                                mb,
                                                                nrhs,
                                                                nnzb,
                                                                (const rocsparse_mat_descr)descrA,
                                                                bsrSortedValA,
                                                                bsrSortedRowPtrA,
                                                                bsrSortedColIndA,
                                                                blockDim,
                                                                (rocsparse_mat_info)info,
                                                                rocsparse_analysis_policy_force,
                                                                rocsparse_solve_policy_auto,
                                                                pBuffer));
}

hipsparseStatus_t hipsparseCbsrsm2_analysis(hipsparseHandle_t         handle,
                                            hipsparseDirection_t      dirA,
                                            hipsparseOperation_t      transA,
                                            hipsparseOperation_t      transX,
                                            int                       mb,
                                            int                       nrhs,
                                            int                       nnzb,
                                            const hipsparseMatDescr_t descrA,
                                            const hipComplex*         bsrSortedValA,
                                            const int*                bsrSortedRowPtrA,
                                            const int*                bsrSortedColIndA,
                                            int                       blockDim,
                                            bsrsm2Info_t              info,
                                            hipsparseSolvePolicy_t    policy,
                                            void*                     pBuffer)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_cbsrsm_analysis((rocsparse_handle)handle,
                                  hipDirectionToHCCDirection(dirA),
                                  hipOperationToHCCOperation(transA),
                                  hipOperationToHCCOperation(transX),
                                  mb,
                                  nrhs,
                                  nnzb,
                                  (const rocsparse_mat_descr)descrA,
                                  (const rocsparse_float_complex*)bsrSortedValA,
                                  bsrSortedRowPtrA,
                                  bsrSortedColIndA,
                                  blockDim,
                                  (rocsparse_mat_info)info,
                                  rocsparse_analysis_policy_force,
                                  rocsparse_solve_policy_auto,
                                  pBuffer));
}

hipsparseStatus_t hipsparseZbsrsm2_analysis(hipsparseHandle_t         handle,
                                            hipsparseDirection_t      dirA,
                                            hipsparseOperation_t      transA,
                                            hipsparseOperation_t      transX,
                                            int                       mb,
                                            int                       nrhs,
                                            int                       nnzb,
                                            const hipsparseMatDescr_t descrA,
                                            const hipDoubleComplex*   bsrSortedValA,
                                            const int*                bsrSortedRowPtrA,
                                            const int*                bsrSortedColIndA,
                                            int                       blockDim,
                                            bsrsm2Info_t              info,
                                            hipsparseSolvePolicy_t    policy,
                                            void*                     pBuffer)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_zbsrsm_analysis((rocsparse_handle)handle,
                                  hipDirectionToHCCDirection(dirA),
                                  hipOperationToHCCOperation(transA),
                                  hipOperationToHCCOperation(transX),
                                  mb,
                                  nrhs,
                                  nnzb,
                                  (const rocsparse_mat_descr)descrA,
                                  (const rocsparse_double_complex*)bsrSortedValA,
                                  bsrSortedRowPtrA,
                                  bsrSortedColIndA,
                                  blockDim,
                                  (rocsparse_mat_info)info,
                                  rocsparse_analysis_policy_force,
                                  rocsparse_solve_policy_auto,
                                  pBuffer));
}

hipsparseStatus_t hipsparseSbsrsm2_solve(hipsparseHandle_t         handle,
                                         hipsparseDirection_t      dirA,
                                         hipsparseOperation_t      transA,
                                         hipsparseOperation_t      transX,
                                         int                       mb,
                                         int                       nrhs,
                                         int                       nnzb,
                                         const float*              alpha,
                                         const hipsparseMatDescr_t descrA,
                                         const float*              bsrSortedValA,
                                         const int*                bsrSortedRowPtrA,
                                         const int*                bsrSortedColIndA,
                                         int                       blockDim,
                                         bsrsm2Info_t              info,
                                         const float*              B,
                                         int                       ldb,
                                         float*                    X,
                                         int                       ldx,
                                         hipsparseSolvePolicy_t    policy,
                                         void*                     pBuffer)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_sbsrsm_solve((rocsparse_handle)handle,
                                                             hipDirectionToHCCDirection(dirA),
                                                             hipOperationToHCCOperation(transA),
                                                             hipOperationToHCCOperation(transX),
                                                             mb,
                                                             nrhs,
                                                             nnzb,
                                                             alpha,
                                                             (const rocsparse_mat_descr)descrA,
                                                             bsrSortedValA,
                                                             bsrSortedRowPtrA,
                                                             bsrSortedColIndA,
                                                             blockDim,
                                                             (rocsparse_mat_info)info,
                                                             B,
                                                             ldb,
                                                             X,
                                                             ldx,
                                                             rocsparse_solve_policy_auto,
                                                             pBuffer));
}

hipsparseStatus_t hipsparseDbsrsm2_solve(hipsparseHandle_t         handle,
                                         hipsparseDirection_t      dirA,
                                         hipsparseOperation_t      transA,
                                         hipsparseOperation_t      transX,
                                         int                       mb,
                                         int                       nrhs,
                                         int                       nnzb,
                                         const double*             alpha,
                                         const hipsparseMatDescr_t descrA,
                                         const double*             bsrSortedValA,
                                         const int*                bsrSortedRowPtrA,
                                         const int*                bsrSortedColIndA,
                                         int                       blockDim,
                                         bsrsm2Info_t              info,
                                         const double*             B,
                                         int                       ldb,
                                         double*                   X,
                                         int                       ldx,
                                         hipsparseSolvePolicy_t    policy,
                                         void*                     pBuffer)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_dbsrsm_solve((rocsparse_handle)handle,
                                                             hipDirectionToHCCDirection(dirA),
                                                             hipOperationToHCCOperation(transA),
                                                             hipOperationToHCCOperation(transX),
                                                             mb,
                                                             nrhs,
                                                             nnzb,
                                                             alpha,
                                                             (const rocsparse_mat_descr)descrA,
                                                             bsrSortedValA,
                                                             bsrSortedRowPtrA,
                                                             bsrSortedColIndA,
                                                             blockDim,
                                                             (rocsparse_mat_info)info,
                                                             B,
                                                             ldb,
                                                             X,
                                                             ldx,
                                                             rocsparse_solve_policy_auto,
                                                             pBuffer));
}

hipsparseStatus_t hipsparseCbsrsm2_solve(hipsparseHandle_t         handle,
                                         hipsparseDirection_t      dirA,
                                         hipsparseOperation_t      transA,
                                         hipsparseOperation_t      transX,
                                         int                       mb,
                                         int                       nrhs,
                                         int                       nnzb,
                                         const hipComplex*         alpha,
                                         const hipsparseMatDescr_t descrA,
                                         const hipComplex*         bsrSortedValA,
                                         const int*                bsrSortedRowPtrA,
                                         const int*                bsrSortedColIndA,
                                         int                       blockDim,
                                         bsrsm2Info_t              info,
                                         const hipComplex*         B,
                                         int                       ldb,
                                         hipComplex*               X,
                                         int                       ldx,
                                         hipsparseSolvePolicy_t    policy,
                                         void*                     pBuffer)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_cbsrsm_solve((rocsparse_handle)handle,
                               hipDirectionToHCCDirection(dirA),
                               hipOperationToHCCOperation(transA),
                               hipOperationToHCCOperation(transX),
                               mb,
                               nrhs,
                               nnzb,
                               (const rocsparse_float_complex*)alpha,
                               (const rocsparse_mat_descr)descrA,
                               (const rocsparse_float_complex*)bsrSortedValA,
                               bsrSortedRowPtrA,
                               bsrSortedColIndA,
                               blockDim,
                               (rocsparse_mat_info)info,
                               (const rocsparse_float_complex*)B,
                               ldb,
                               (rocsparse_float_complex*)X,
                               ldx,
                               rocsparse_solve_policy_auto,
                               pBuffer));
}

hipsparseStatus_t hipsparseZbsrsm2_solve(hipsparseHandle_t         handle,
                                         hipsparseDirection_t      dirA,
                                         hipsparseOperation_t      transA,
                                         hipsparseOperation_t      transX,
                                         int                       mb,
                                         int                       nrhs,
                                         int                       nnzb,
                                         const hipDoubleComplex*   alpha,
                                         const hipsparseMatDescr_t descrA,
                                         const hipDoubleComplex*   bsrSortedValA,
                                         const int*                bsrSortedRowPtrA,
                                         const int*                bsrSortedColIndA,
                                         int                       blockDim,
                                         bsrsm2Info_t              info,
                                         const hipDoubleComplex*   B,
                                         int                       ldb,
                                         hipDoubleComplex*         X,
                                         int                       ldx,
                                         hipsparseSolvePolicy_t    policy,
                                         void*                     pBuffer)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_zbsrsm_solve((rocsparse_handle)handle,
                               hipDirectionToHCCDirection(dirA),
                               hipOperationToHCCOperation(transA),
                               hipOperationToHCCOperation(transX),
                               mb,
                               nrhs,
                               nnzb,
                               (const rocsparse_double_complex*)alpha,
                               (const rocsparse_mat_descr)descrA,
                               (const rocsparse_double_complex*)bsrSortedValA,
                               bsrSortedRowPtrA,
                               bsrSortedColIndA,
                               blockDim,
                               (rocsparse_mat_info)info,
                               (const rocsparse_double_complex*)B,
                               ldb,
                               (rocsparse_double_complex*)X,
                               ldx,
                               rocsparse_solve_policy_auto,
                               pBuffer));
}

hipsparseStatus_t
    hipsparseXcsrsm2_zeroPivot(hipsparseHandle_t handle, csrsm2Info_t info, int* position)
{
    // Obtain stream, to explicitly sync (cusparse csrsm2_zeropivot is blocking)
    hipStream_t stream;
    RETURN_IF_HIPSPARSE_ERROR(hipsparseGetStream(handle, &stream));

    // csrsm zero pivot
    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse_csrsm_zero_pivot((rocsparse_handle)handle, (rocsparse_mat_info)info, position));

    // Synchronize stream
    RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseScsrsm2_bufferSizeExt(hipsparseHandle_t         handle,
                                                 int                       algo,
                                                 hipsparseOperation_t      transA,
                                                 hipsparseOperation_t      transB,
                                                 int                       m,
                                                 int                       nrhs,
                                                 int                       nnz,
                                                 const float*              alpha,
                                                 const hipsparseMatDescr_t descrA,
                                                 const float*              csrSortedValA,
                                                 const int*                csrSortedRowPtrA,
                                                 const int*                csrSortedColIndA,
                                                 const float*              B,
                                                 int                       ldb,
                                                 csrsm2Info_t              info,
                                                 hipsparseSolvePolicy_t    policy,
                                                 size_t*                   pBufferSize)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_scsrsm_buffer_size((rocsparse_handle)handle,
                                     hipOperationToHCCOperation(transA),
                                     hipOperationToHCCOperation(transB),
                                     m,
                                     nrhs,
                                     nnz,
                                     alpha,
                                     (const rocsparse_mat_descr)descrA,
                                     csrSortedValA,
                                     csrSortedRowPtrA,
                                     csrSortedColIndA,
                                     B,
                                     ldb,
                                     (rocsparse_mat_info)info,
                                     rocsparse_solve_policy_auto,
                                     pBufferSize));
}

hipsparseStatus_t hipsparseDcsrsm2_bufferSizeExt(hipsparseHandle_t         handle,
                                                 int                       algo,
                                                 hipsparseOperation_t      transA,
                                                 hipsparseOperation_t      transB,
                                                 int                       m,
                                                 int                       nrhs,
                                                 int                       nnz,
                                                 const double*             alpha,
                                                 const hipsparseMatDescr_t descrA,
                                                 const double*             csrSortedValA,
                                                 const int*                csrSortedRowPtrA,
                                                 const int*                csrSortedColIndA,
                                                 const double*             B,
                                                 int                       ldb,
                                                 csrsm2Info_t              info,
                                                 hipsparseSolvePolicy_t    policy,
                                                 size_t*                   pBufferSize)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_dcsrsm_buffer_size((rocsparse_handle)handle,
                                     hipOperationToHCCOperation(transA),
                                     hipOperationToHCCOperation(transB),
                                     m,
                                     nrhs,
                                     nnz,
                                     alpha,
                                     (const rocsparse_mat_descr)descrA,
                                     csrSortedValA,
                                     csrSortedRowPtrA,
                                     csrSortedColIndA,
                                     B,
                                     ldb,
                                     (rocsparse_mat_info)info,
                                     rocsparse_solve_policy_auto,
                                     pBufferSize));
}

hipsparseStatus_t hipsparseCcsrsm2_bufferSizeExt(hipsparseHandle_t         handle,
                                                 int                       algo,
                                                 hipsparseOperation_t      transA,
                                                 hipsparseOperation_t      transB,
                                                 int                       m,
                                                 int                       nrhs,
                                                 int                       nnz,
                                                 const hipComplex*         alpha,
                                                 const hipsparseMatDescr_t descrA,
                                                 const hipComplex*         csrSortedValA,
                                                 const int*                csrSortedRowPtrA,
                                                 const int*                csrSortedColIndA,
                                                 const hipComplex*         B,
                                                 int                       ldb,
                                                 csrsm2Info_t              info,
                                                 hipsparseSolvePolicy_t    policy,
                                                 size_t*                   pBufferSize)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_ccsrsm_buffer_size((rocsparse_handle)handle,
                                     hipOperationToHCCOperation(transA),
                                     hipOperationToHCCOperation(transB),
                                     m,
                                     nrhs,
                                     nnz,
                                     (const rocsparse_float_complex*)alpha,
                                     (const rocsparse_mat_descr)descrA,
                                     (const rocsparse_float_complex*)csrSortedValA,
                                     csrSortedRowPtrA,
                                     csrSortedColIndA,
                                     (const rocsparse_float_complex*)B,
                                     ldb,
                                     (rocsparse_mat_info)info,
                                     rocsparse_solve_policy_auto,
                                     pBufferSize));
}

hipsparseStatus_t hipsparseZcsrsm2_bufferSizeExt(hipsparseHandle_t         handle,
                                                 int                       algo,
                                                 hipsparseOperation_t      transA,
                                                 hipsparseOperation_t      transB,
                                                 int                       m,
                                                 int                       nrhs,
                                                 int                       nnz,
                                                 const hipDoubleComplex*   alpha,
                                                 const hipsparseMatDescr_t descrA,
                                                 const hipDoubleComplex*   csrSortedValA,
                                                 const int*                csrSortedRowPtrA,
                                                 const int*                csrSortedColIndA,
                                                 const hipDoubleComplex*   B,
                                                 int                       ldb,
                                                 csrsm2Info_t              info,
                                                 hipsparseSolvePolicy_t    policy,
                                                 size_t*                   pBufferSize)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_zcsrsm_buffer_size((rocsparse_handle)handle,
                                     hipOperationToHCCOperation(transA),
                                     hipOperationToHCCOperation(transB),
                                     m,
                                     nrhs,
                                     nnz,
                                     (const rocsparse_double_complex*)alpha,
                                     (const rocsparse_mat_descr)descrA,
                                     (const rocsparse_double_complex*)csrSortedValA,
                                     csrSortedRowPtrA,
                                     csrSortedColIndA,
                                     (const rocsparse_double_complex*)B,
                                     ldb,
                                     (rocsparse_mat_info)info,
                                     rocsparse_solve_policy_auto,
                                     pBufferSize));
}

hipsparseStatus_t hipsparseScsrsm2_analysis(hipsparseHandle_t         handle,
                                            int                       algo,
                                            hipsparseOperation_t      transA,
                                            hipsparseOperation_t      transB,
                                            int                       m,
                                            int                       nrhs,
                                            int                       nnz,
                                            const float*              alpha,
                                            const hipsparseMatDescr_t descrA,
                                            const float*              csrSortedValA,
                                            const int*                csrSortedRowPtrA,
                                            const int*                csrSortedColIndA,
                                            const float*              B,
                                            int                       ldb,
                                            csrsm2Info_t              info,
                                            hipsparseSolvePolicy_t    policy,
                                            void*                     pBuffer)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_scsrsm_analysis((rocsparse_handle)handle,
                                                                hipOperationToHCCOperation(transA),
                                                                hipOperationToHCCOperation(transB),
                                                                m,
                                                                nrhs,
                                                                nnz,
                                                                alpha,
                                                                (const rocsparse_mat_descr)descrA,
                                                                csrSortedValA,
                                                                csrSortedRowPtrA,
                                                                csrSortedColIndA,
                                                                B,
                                                                ldb,
                                                                (rocsparse_mat_info)info,
                                                                rocsparse_analysis_policy_force,
                                                                rocsparse_solve_policy_auto,
                                                                pBuffer));
}

hipsparseStatus_t hipsparseDcsrsm2_analysis(hipsparseHandle_t         handle,
                                            int                       algo,
                                            hipsparseOperation_t      transA,
                                            hipsparseOperation_t      transB,
                                            int                       m,
                                            int                       nrhs,
                                            int                       nnz,
                                            const double*             alpha,
                                            const hipsparseMatDescr_t descrA,
                                            const double*             csrSortedValA,
                                            const int*                csrSortedRowPtrA,
                                            const int*                csrSortedColIndA,
                                            const double*             B,
                                            int                       ldb,
                                            csrsm2Info_t              info,
                                            hipsparseSolvePolicy_t    policy,
                                            void*                     pBuffer)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_dcsrsm_analysis((rocsparse_handle)handle,
                                                                hipOperationToHCCOperation(transA),
                                                                hipOperationToHCCOperation(transB),
                                                                m,
                                                                nrhs,
                                                                nnz,
                                                                alpha,
                                                                (const rocsparse_mat_descr)descrA,
                                                                csrSortedValA,
                                                                csrSortedRowPtrA,
                                                                csrSortedColIndA,
                                                                B,
                                                                ldb,
                                                                (rocsparse_mat_info)info,
                                                                rocsparse_analysis_policy_force,
                                                                rocsparse_solve_policy_auto,
                                                                pBuffer));
}

hipsparseStatus_t hipsparseCcsrsm2_analysis(hipsparseHandle_t         handle,
                                            int                       algo,
                                            hipsparseOperation_t      transA,
                                            hipsparseOperation_t      transB,
                                            int                       m,
                                            int                       nrhs,
                                            int                       nnz,
                                            const hipComplex*         alpha,
                                            const hipsparseMatDescr_t descrA,
                                            const hipComplex*         csrSortedValA,
                                            const int*                csrSortedRowPtrA,
                                            const int*                csrSortedColIndA,
                                            const hipComplex*         B,
                                            int                       ldb,
                                            csrsm2Info_t              info,
                                            hipsparseSolvePolicy_t    policy,
                                            void*                     pBuffer)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_ccsrsm_analysis((rocsparse_handle)handle,
                                  hipOperationToHCCOperation(transA),
                                  hipOperationToHCCOperation(transB),
                                  m,
                                  nrhs,
                                  nnz,
                                  (const rocsparse_float_complex*)alpha,
                                  (const rocsparse_mat_descr)descrA,
                                  (const rocsparse_float_complex*)csrSortedValA,
                                  csrSortedRowPtrA,
                                  csrSortedColIndA,
                                  (const rocsparse_float_complex*)B,
                                  ldb,
                                  (rocsparse_mat_info)info,
                                  rocsparse_analysis_policy_force,
                                  rocsparse_solve_policy_auto,
                                  pBuffer));
}

hipsparseStatus_t hipsparseZcsrsm2_analysis(hipsparseHandle_t         handle,
                                            int                       algo,
                                            hipsparseOperation_t      transA,
                                            hipsparseOperation_t      transB,
                                            int                       m,
                                            int                       nrhs,
                                            int                       nnz,
                                            const hipDoubleComplex*   alpha,
                                            const hipsparseMatDescr_t descrA,
                                            const hipDoubleComplex*   csrSortedValA,
                                            const int*                csrSortedRowPtrA,
                                            const int*                csrSortedColIndA,
                                            const hipDoubleComplex*   B,
                                            int                       ldb,
                                            csrsm2Info_t              info,
                                            hipsparseSolvePolicy_t    policy,
                                            void*                     pBuffer)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_zcsrsm_analysis((rocsparse_handle)handle,
                                  hipOperationToHCCOperation(transA),
                                  hipOperationToHCCOperation(transB),
                                  m,
                                  nrhs,
                                  nnz,
                                  (const rocsparse_double_complex*)alpha,
                                  (const rocsparse_mat_descr)descrA,
                                  (const rocsparse_double_complex*)csrSortedValA,
                                  csrSortedRowPtrA,
                                  csrSortedColIndA,
                                  (const rocsparse_double_complex*)B,
                                  ldb,
                                  (rocsparse_mat_info)info,
                                  rocsparse_analysis_policy_force,
                                  rocsparse_solve_policy_auto,
                                  pBuffer));
}

hipsparseStatus_t hipsparseScsrsm2_solve(hipsparseHandle_t         handle,
                                         int                       algo,
                                         hipsparseOperation_t      transA,
                                         hipsparseOperation_t      transB,
                                         int                       m,
                                         int                       nrhs,
                                         int                       nnz,
                                         const float*              alpha,
                                         const hipsparseMatDescr_t descrA,
                                         const float*              csrSortedValA,
                                         const int*                csrSortedRowPtrA,
                                         const int*                csrSortedColIndA,
                                         float*                    B,
                                         int                       ldb,
                                         csrsm2Info_t              info,
                                         hipsparseSolvePolicy_t    policy,
                                         void*                     pBuffer)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_scsrsm_solve((rocsparse_handle)handle,
                                                             hipOperationToHCCOperation(transA),
                                                             hipOperationToHCCOperation(transB),
                                                             m,
                                                             nrhs,
                                                             nnz,
                                                             alpha,
                                                             (const rocsparse_mat_descr)descrA,
                                                             csrSortedValA,
                                                             csrSortedRowPtrA,
                                                             csrSortedColIndA,
                                                             B,
                                                             ldb,
                                                             (rocsparse_mat_info)info,
                                                             rocsparse_solve_policy_auto,
                                                             pBuffer));
}

hipsparseStatus_t hipsparseDcsrsm2_solve(hipsparseHandle_t         handle,
                                         int                       algo,
                                         hipsparseOperation_t      transA,
                                         hipsparseOperation_t      transB,
                                         int                       m,
                                         int                       nrhs,
                                         int                       nnz,
                                         const double*             alpha,
                                         const hipsparseMatDescr_t descrA,
                                         const double*             csrSortedValA,
                                         const int*                csrSortedRowPtrA,
                                         const int*                csrSortedColIndA,
                                         double*                   B,
                                         int                       ldb,
                                         csrsm2Info_t              info,
                                         hipsparseSolvePolicy_t    policy,
                                         void*                     pBuffer)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_dcsrsm_solve((rocsparse_handle)handle,
                                                             hipOperationToHCCOperation(transA),
                                                             hipOperationToHCCOperation(transB),
                                                             m,
                                                             nrhs,
                                                             nnz,
                                                             alpha,
                                                             (const rocsparse_mat_descr)descrA,
                                                             csrSortedValA,
                                                             csrSortedRowPtrA,
                                                             csrSortedColIndA,
                                                             B,
                                                             ldb,
                                                             (rocsparse_mat_info)info,
                                                             rocsparse_solve_policy_auto,
                                                             pBuffer));
}

hipsparseStatus_t hipsparseCcsrsm2_solve(hipsparseHandle_t         handle,
                                         int                       algo,
                                         hipsparseOperation_t      transA,
                                         hipsparseOperation_t      transB,
                                         int                       m,
                                         int                       nrhs,
                                         int                       nnz,
                                         const hipComplex*         alpha,
                                         const hipsparseMatDescr_t descrA,
                                         const hipComplex*         csrSortedValA,
                                         const int*                csrSortedRowPtrA,
                                         const int*                csrSortedColIndA,
                                         hipComplex*               B,
                                         int                       ldb,
                                         csrsm2Info_t              info,
                                         hipsparseSolvePolicy_t    policy,
                                         void*                     pBuffer)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_ccsrsm_solve((rocsparse_handle)handle,
                               hipOperationToHCCOperation(transA),
                               hipOperationToHCCOperation(transB),
                               m,
                               nrhs,
                               nnz,
                               (const rocsparse_float_complex*)alpha,
                               (const rocsparse_mat_descr)descrA,
                               (const rocsparse_float_complex*)csrSortedValA,
                               csrSortedRowPtrA,
                               csrSortedColIndA,
                               (rocsparse_float_complex*)B,
                               ldb,
                               (rocsparse_mat_info)info,
                               rocsparse_solve_policy_auto,
                               pBuffer));
}

hipsparseStatus_t hipsparseZcsrsm2_solve(hipsparseHandle_t         handle,
                                         int                       algo,
                                         hipsparseOperation_t      transA,
                                         hipsparseOperation_t      transB,
                                         int                       m,
                                         int                       nrhs,
                                         int                       nnz,
                                         const hipDoubleComplex*   alpha,
                                         const hipsparseMatDescr_t descrA,
                                         const hipDoubleComplex*   csrSortedValA,
                                         const int*                csrSortedRowPtrA,
                                         const int*                csrSortedColIndA,
                                         hipDoubleComplex*         B,
                                         int                       ldb,
                                         csrsm2Info_t              info,
                                         hipsparseSolvePolicy_t    policy,
                                         void*                     pBuffer)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_zcsrsm_solve((rocsparse_handle)handle,
                               hipOperationToHCCOperation(transA),
                               hipOperationToHCCOperation(transB),
                               m,
                               nrhs,
                               nnz,
                               (const rocsparse_double_complex*)alpha,
                               (const rocsparse_mat_descr)descrA,
                               (const rocsparse_double_complex*)csrSortedValA,
                               csrSortedRowPtrA,
                               csrSortedColIndA,
                               (rocsparse_double_complex*)B,
                               ldb,
                               (rocsparse_mat_info)info,
                               rocsparse_solve_policy_auto,
                               pBuffer));
}

hipsparseStatus_t hipsparseSgemmi(hipsparseHandle_t handle,
                                  int               m,
                                  int               n,
                                  int               k,
                                  int               nnz,
                                  const float*      alpha,
                                  const float*      A,
                                  int               lda,
                                  const float*      cscValB,
                                  const int*        cscColPtrB,
                                  const int*        cscRowIndB,
                                  const float*      beta,
                                  float*            C,
                                  int               ldc)
{
    rocsparse_mat_descr descr;
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_create_mat_descr(&descr));
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_sgemmi((rocsparse_handle)handle,
                                               rocsparse_operation_none,
                                               rocsparse_operation_transpose,
                                               m,
                                               n,
                                               k,
                                               nnz,
                                               alpha,
                                               A,
                                               lda,
                                               descr,
                                               cscValB,
                                               cscColPtrB,
                                               cscRowIndB,
                                               beta,
                                               C,
                                               ldc));
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_destroy_mat_descr(descr));

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseDgemmi(hipsparseHandle_t handle,
                                  int               m,
                                  int               n,
                                  int               k,
                                  int               nnz,
                                  const double*     alpha,
                                  const double*     A,
                                  int               lda,
                                  const double*     cscValB,
                                  const int*        cscColPtrB,
                                  const int*        cscRowIndB,
                                  const double*     beta,
                                  double*           C,
                                  int               ldc)
{
    rocsparse_mat_descr descr;
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_create_mat_descr(&descr));
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_dgemmi((rocsparse_handle)handle,
                                               rocsparse_operation_none,
                                               rocsparse_operation_transpose,
                                               m,
                                               n,
                                               k,
                                               nnz,
                                               alpha,
                                               A,
                                               lda,
                                               descr,
                                               cscValB,
                                               cscColPtrB,
                                               cscRowIndB,
                                               beta,
                                               C,
                                               ldc));
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_destroy_mat_descr(descr));

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseCgemmi(hipsparseHandle_t handle,
                                  int               m,
                                  int               n,
                                  int               k,
                                  int               nnz,
                                  const hipComplex* alpha,
                                  const hipComplex* A,
                                  int               lda,
                                  const hipComplex* cscValB,
                                  const int*        cscColPtrB,
                                  const int*        cscRowIndB,
                                  const hipComplex* beta,
                                  hipComplex*       C,
                                  int               ldc)
{
    rocsparse_mat_descr descr;
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_create_mat_descr(&descr));
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_cgemmi((rocsparse_handle)handle,
                                               rocsparse_operation_none,
                                               rocsparse_operation_transpose,
                                               m,
                                               n,
                                               k,
                                               nnz,
                                               (const rocsparse_float_complex*)alpha,
                                               (const rocsparse_float_complex*)A,
                                               lda,
                                               descr,
                                               (const rocsparse_float_complex*)cscValB,
                                               cscColPtrB,
                                               cscRowIndB,
                                               (const rocsparse_float_complex*)beta,
                                               (rocsparse_float_complex*)C,
                                               ldc));
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_destroy_mat_descr(descr));

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseZgemmi(hipsparseHandle_t       handle,
                                  int                     m,
                                  int                     n,
                                  int                     k,
                                  int                     nnz,
                                  const hipDoubleComplex* alpha,
                                  const hipDoubleComplex* A,
                                  int                     lda,
                                  const hipDoubleComplex* cscValB,
                                  const int*              cscColPtrB,
                                  const int*              cscRowIndB,
                                  const hipDoubleComplex* beta,
                                  hipDoubleComplex*       C,
                                  int                     ldc)
{
    rocsparse_mat_descr descr;
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_create_mat_descr(&descr));
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_zgemmi((rocsparse_handle)handle,
                                               rocsparse_operation_none,
                                               rocsparse_operation_transpose,
                                               m,
                                               n,
                                               k,
                                               nnz,
                                               (const rocsparse_double_complex*)alpha,
                                               (const rocsparse_double_complex*)A,
                                               lda,
                                               descr,
                                               (const rocsparse_double_complex*)cscValB,
                                               cscColPtrB,
                                               cscRowIndB,
                                               (const rocsparse_double_complex*)beta,
                                               (rocsparse_double_complex*)C,
                                               ldc));
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_destroy_mat_descr(descr));

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseXcsrgeamNnz(hipsparseHandle_t         handle,
                                       int                       m,
                                       int                       n,
                                       const hipsparseMatDescr_t descrA,
                                       int                       nnzA,
                                       const int*                csrRowPtrA,
                                       const int*                csrColIndA,
                                       const hipsparseMatDescr_t descrB,
                                       int                       nnzB,
                                       const int*                csrRowPtrB,
                                       const int*                csrColIndB,
                                       const hipsparseMatDescr_t descrC,
                                       int*                      csrRowPtrC,
                                       int*                      nnzTotalDevHostPtr)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_csrgeam_nnz((rocsparse_handle)handle,
                                                            m,
                                                            n,
                                                            (const rocsparse_mat_descr)descrA,
                                                            nnzA,
                                                            csrRowPtrA,
                                                            csrColIndA,
                                                            (const rocsparse_mat_descr)descrB,
                                                            nnzB,
                                                            csrRowPtrB,
                                                            csrColIndB,
                                                            (const rocsparse_mat_descr)descrC,
                                                            csrRowPtrC,
                                                            nnzTotalDevHostPtr));
}

hipsparseStatus_t hipsparseScsrgeam(hipsparseHandle_t         handle,
                                    int                       m,
                                    int                       n,
                                    const float*              alpha,
                                    const hipsparseMatDescr_t descrA,
                                    int                       nnzA,
                                    const float*              csrValA,
                                    const int*                csrRowPtrA,
                                    const int*                csrColIndA,
                                    const float*              beta,
                                    const hipsparseMatDescr_t descrB,
                                    int                       nnzB,
                                    const float*              csrValB,
                                    const int*                csrRowPtrB,
                                    const int*                csrColIndB,
                                    const hipsparseMatDescr_t descrC,
                                    float*                    csrValC,
                                    int*                      csrRowPtrC,
                                    int*                      csrColIndC)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_scsrgeam((rocsparse_handle)handle,
                                                         m,
                                                         n,
                                                         alpha,
                                                         (const rocsparse_mat_descr)descrA,
                                                         nnzA,
                                                         csrValA,
                                                         csrRowPtrA,
                                                         csrColIndA,
                                                         beta,
                                                         (const rocsparse_mat_descr)descrB,
                                                         nnzB,
                                                         csrValB,
                                                         csrRowPtrB,
                                                         csrColIndB,
                                                         (const rocsparse_mat_descr)descrC,
                                                         csrValC,
                                                         csrRowPtrC,
                                                         csrColIndC));
}

hipsparseStatus_t hipsparseDcsrgeam(hipsparseHandle_t         handle,
                                    int                       m,
                                    int                       n,
                                    const double*             alpha,
                                    const hipsparseMatDescr_t descrA,
                                    int                       nnzA,
                                    const double*             csrValA,
                                    const int*                csrRowPtrA,
                                    const int*                csrColIndA,
                                    const double*             beta,
                                    const hipsparseMatDescr_t descrB,
                                    int                       nnzB,
                                    const double*             csrValB,
                                    const int*                csrRowPtrB,
                                    const int*                csrColIndB,
                                    const hipsparseMatDescr_t descrC,
                                    double*                   csrValC,
                                    int*                      csrRowPtrC,
                                    int*                      csrColIndC)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_dcsrgeam((rocsparse_handle)handle,
                                                         m,
                                                         n,
                                                         alpha,
                                                         (const rocsparse_mat_descr)descrA,
                                                         nnzA,
                                                         csrValA,
                                                         csrRowPtrA,
                                                         csrColIndA,
                                                         beta,
                                                         (const rocsparse_mat_descr)descrB,
                                                         nnzB,
                                                         csrValB,
                                                         csrRowPtrB,
                                                         csrColIndB,
                                                         (const rocsparse_mat_descr)descrC,
                                                         csrValC,
                                                         csrRowPtrC,
                                                         csrColIndC));
}

hipsparseStatus_t hipsparseCcsrgeam(hipsparseHandle_t         handle,
                                    int                       m,
                                    int                       n,
                                    const hipComplex*         alpha,
                                    const hipsparseMatDescr_t descrA,
                                    int                       nnzA,
                                    const hipComplex*         csrValA,
                                    const int*                csrRowPtrA,
                                    const int*                csrColIndA,
                                    const hipComplex*         beta,
                                    const hipsparseMatDescr_t descrB,
                                    int                       nnzB,
                                    const hipComplex*         csrValB,
                                    const int*                csrRowPtrB,
                                    const int*                csrColIndB,
                                    const hipsparseMatDescr_t descrC,
                                    hipComplex*               csrValC,
                                    int*                      csrRowPtrC,
                                    int*                      csrColIndC)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_ccsrgeam((rocsparse_handle)handle,
                                                         m,
                                                         n,
                                                         (const rocsparse_float_complex*)alpha,
                                                         (const rocsparse_mat_descr)descrA,
                                                         nnzA,
                                                         (const rocsparse_float_complex*)csrValA,
                                                         csrRowPtrA,
                                                         csrColIndA,
                                                         (const rocsparse_float_complex*)beta,
                                                         (const rocsparse_mat_descr)descrB,
                                                         nnzB,
                                                         (const rocsparse_float_complex*)csrValB,
                                                         csrRowPtrB,
                                                         csrColIndB,
                                                         (const rocsparse_mat_descr)descrC,
                                                         (rocsparse_float_complex*)csrValC,
                                                         csrRowPtrC,
                                                         csrColIndC));
}

hipsparseStatus_t hipsparseZcsrgeam(hipsparseHandle_t         handle,
                                    int                       m,
                                    int                       n,
                                    const hipDoubleComplex*   alpha,
                                    const hipsparseMatDescr_t descrA,
                                    int                       nnzA,
                                    const hipDoubleComplex*   csrValA,
                                    const int*                csrRowPtrA,
                                    const int*                csrColIndA,
                                    const hipDoubleComplex*   beta,
                                    const hipsparseMatDescr_t descrB,
                                    int                       nnzB,
                                    const hipDoubleComplex*   csrValB,
                                    const int*                csrRowPtrB,
                                    const int*                csrColIndB,
                                    const hipsparseMatDescr_t descrC,
                                    hipDoubleComplex*         csrValC,
                                    int*                      csrRowPtrC,
                                    int*                      csrColIndC)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_zcsrgeam((rocsparse_handle)handle,
                                                         m,
                                                         n,
                                                         (const rocsparse_double_complex*)alpha,
                                                         (const rocsparse_mat_descr)descrA,
                                                         nnzA,
                                                         (const rocsparse_double_complex*)csrValA,
                                                         csrRowPtrA,
                                                         csrColIndA,
                                                         (const rocsparse_double_complex*)beta,
                                                         (const rocsparse_mat_descr)descrB,
                                                         nnzB,
                                                         (const rocsparse_double_complex*)csrValB,
                                                         csrRowPtrB,
                                                         csrColIndB,
                                                         (const rocsparse_mat_descr)descrC,
                                                         (rocsparse_double_complex*)csrValC,
                                                         csrRowPtrC,
                                                         csrColIndC));
}

hipsparseStatus_t hipsparseScsrgeam2_bufferSizeExt(hipsparseHandle_t         handle,
                                                   int                       m,
                                                   int                       n,
                                                   const float*              alpha,
                                                   const hipsparseMatDescr_t descrA,
                                                   int                       nnzA,
                                                   const float*              csrSortedValA,
                                                   const int*                csrSortedRowPtrA,
                                                   const int*                csrSortedColIndA,
                                                   const float*              beta,
                                                   const hipsparseMatDescr_t descrB,
                                                   int                       nnzB,
                                                   const float*              csrSortedValB,
                                                   const int*                csrSortedRowPtrB,
                                                   const int*                csrSortedColIndB,
                                                   const hipsparseMatDescr_t descrC,
                                                   const float*              csrSortedValC,
                                                   const int*                csrSortedRowPtrC,
                                                   const int*                csrSortedColIndC,
                                                   size_t*                   pBufferSizeInBytes)
{
    *pBufferSizeInBytes = 4;

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseDcsrgeam2_bufferSizeExt(hipsparseHandle_t         handle,
                                                   int                       m,
                                                   int                       n,
                                                   const double*             alpha,
                                                   const hipsparseMatDescr_t descrA,
                                                   int                       nnzA,
                                                   const double*             csrSortedValA,
                                                   const int*                csrSortedRowPtrA,
                                                   const int*                csrSortedColIndA,
                                                   const double*             beta,
                                                   const hipsparseMatDescr_t descrB,
                                                   int                       nnzB,
                                                   const double*             csrSortedValB,
                                                   const int*                csrSortedRowPtrB,
                                                   const int*                csrSortedColIndB,
                                                   const hipsparseMatDescr_t descrC,
                                                   const double*             csrSortedValC,
                                                   const int*                csrSortedRowPtrC,
                                                   const int*                csrSortedColIndC,
                                                   size_t*                   pBufferSizeInBytes)
{
    *pBufferSizeInBytes = 4;

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseCcsrgeam2_bufferSizeExt(hipsparseHandle_t         handle,
                                                   int                       m,
                                                   int                       n,
                                                   const hipComplex*         alpha,
                                                   const hipsparseMatDescr_t descrA,
                                                   int                       nnzA,
                                                   const hipComplex*         csrSortedValA,
                                                   const int*                csrSortedRowPtrA,
                                                   const int*                csrSortedColIndA,
                                                   const hipComplex*         beta,
                                                   const hipsparseMatDescr_t descrB,
                                                   int                       nnzB,
                                                   const hipComplex*         csrSortedValB,
                                                   const int*                csrSortedRowPtrB,
                                                   const int*                csrSortedColIndB,
                                                   const hipsparseMatDescr_t descrC,
                                                   const hipComplex*         csrSortedValC,
                                                   const int*                csrSortedRowPtrC,
                                                   const int*                csrSortedColIndC,
                                                   size_t*                   pBufferSizeInBytes)
{
    *pBufferSizeInBytes = 4;

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseZcsrgeam2_bufferSizeExt(hipsparseHandle_t         handle,
                                                   int                       m,
                                                   int                       n,
                                                   const hipDoubleComplex*   alpha,
                                                   const hipsparseMatDescr_t descrA,
                                                   int                       nnzA,
                                                   const hipDoubleComplex*   csrSortedValA,
                                                   const int*                csrSortedRowPtrA,
                                                   const int*                csrSortedColIndA,
                                                   const hipDoubleComplex*   beta,
                                                   const hipsparseMatDescr_t descrB,
                                                   int                       nnzB,
                                                   const hipDoubleComplex*   csrSortedValB,
                                                   const int*                csrSortedRowPtrB,
                                                   const int*                csrSortedColIndB,
                                                   const hipsparseMatDescr_t descrC,
                                                   const hipDoubleComplex*   csrSortedValC,
                                                   const int*                csrSortedRowPtrC,
                                                   const int*                csrSortedColIndC,
                                                   size_t*                   pBufferSizeInBytes)
{
    *pBufferSizeInBytes = 4;

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseXcsrgeam2Nnz(hipsparseHandle_t         handle,
                                        int                       m,
                                        int                       n,
                                        const hipsparseMatDescr_t descrA,
                                        int                       nnzA,
                                        const int*                csrSortedRowPtrA,
                                        const int*                csrSortedColIndA,
                                        const hipsparseMatDescr_t descrB,
                                        int                       nnzB,
                                        const int*                csrSortedRowPtrB,
                                        const int*                csrSortedColIndB,
                                        const hipsparseMatDescr_t descrC,
                                        int*                      csrSortedRowPtrC,
                                        int*                      nnzTotalDevHostPtr,
                                        void*                     workspace)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_csrgeam_nnz((rocsparse_handle)handle,
                                                            m,
                                                            n,
                                                            (const rocsparse_mat_descr)descrA,
                                                            nnzA,
                                                            csrSortedRowPtrA,
                                                            csrSortedColIndA,
                                                            (const rocsparse_mat_descr)descrB,
                                                            nnzB,
                                                            csrSortedRowPtrB,
                                                            csrSortedColIndB,
                                                            (const rocsparse_mat_descr)descrC,
                                                            csrSortedRowPtrC,
                                                            nnzTotalDevHostPtr));
}

hipsparseStatus_t hipsparseScsrgeam2(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       n,
                                     const float*              alpha,
                                     const hipsparseMatDescr_t descrA,
                                     int                       nnzA,
                                     const float*              csrSortedValA,
                                     const int*                csrSortedRowPtrA,
                                     const int*                csrSortedColIndA,
                                     const float*              beta,
                                     const hipsparseMatDescr_t descrB,
                                     int                       nnzB,
                                     const float*              csrSortedValB,
                                     const int*                csrSortedRowPtrB,
                                     const int*                csrSortedColIndB,
                                     const hipsparseMatDescr_t descrC,
                                     float*                    csrSortedValC,
                                     int*                      csrSortedRowPtrC,
                                     int*                      csrSortedColIndC,
                                     void*                     pBuffer)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_scsrgeam((rocsparse_handle)handle,
                                                         m,
                                                         n,
                                                         alpha,
                                                         (const rocsparse_mat_descr)descrA,
                                                         nnzA,
                                                         csrSortedValA,
                                                         csrSortedRowPtrA,
                                                         csrSortedColIndA,
                                                         beta,
                                                         (const rocsparse_mat_descr)descrB,
                                                         nnzB,
                                                         csrSortedValB,
                                                         csrSortedRowPtrB,
                                                         csrSortedColIndB,
                                                         (const rocsparse_mat_descr)descrC,
                                                         csrSortedValC,
                                                         csrSortedRowPtrC,
                                                         csrSortedColIndC));
}

hipsparseStatus_t hipsparseDcsrgeam2(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       n,
                                     const double*             alpha,
                                     const hipsparseMatDescr_t descrA,
                                     int                       nnzA,
                                     const double*             csrSortedValA,
                                     const int*                csrSortedRowPtrA,
                                     const int*                csrSortedColIndA,
                                     const double*             beta,
                                     const hipsparseMatDescr_t descrB,
                                     int                       nnzB,
                                     const double*             csrSortedValB,
                                     const int*                csrSortedRowPtrB,
                                     const int*                csrSortedColIndB,
                                     const hipsparseMatDescr_t descrC,
                                     double*                   csrSortedValC,
                                     int*                      csrSortedRowPtrC,
                                     int*                      csrSortedColIndC,
                                     void*                     pBuffer)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_dcsrgeam((rocsparse_handle)handle,
                                                         m,
                                                         n,
                                                         alpha,
                                                         (const rocsparse_mat_descr)descrA,
                                                         nnzA,
                                                         csrSortedValA,
                                                         csrSortedRowPtrA,
                                                         csrSortedColIndA,
                                                         beta,
                                                         (const rocsparse_mat_descr)descrB,
                                                         nnzB,
                                                         csrSortedValB,
                                                         csrSortedRowPtrB,
                                                         csrSortedColIndB,
                                                         (const rocsparse_mat_descr)descrC,
                                                         csrSortedValC,
                                                         csrSortedRowPtrC,
                                                         csrSortedColIndC));
}

hipsparseStatus_t hipsparseCcsrgeam2(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       n,
                                     const hipComplex*         alpha,
                                     const hipsparseMatDescr_t descrA,
                                     int                       nnzA,
                                     const hipComplex*         csrSortedValA,
                                     const int*                csrSortedRowPtrA,
                                     const int*                csrSortedColIndA,
                                     const hipComplex*         beta,
                                     const hipsparseMatDescr_t descrB,
                                     int                       nnzB,
                                     const hipComplex*         csrSortedValB,
                                     const int*                csrSortedRowPtrB,
                                     const int*                csrSortedColIndB,
                                     const hipsparseMatDescr_t descrC,
                                     hipComplex*               csrSortedValC,
                                     int*                      csrSortedRowPtrC,
                                     int*                      csrSortedColIndC,
                                     void*                     pBuffer)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_ccsrgeam((rocsparse_handle)handle,
                           m,
                           n,
                           (const rocsparse_float_complex*)alpha,
                           (const rocsparse_mat_descr)descrA,
                           nnzA,
                           (const rocsparse_float_complex*)csrSortedValA,
                           csrSortedRowPtrA,
                           csrSortedColIndA,
                           (const rocsparse_float_complex*)beta,
                           (const rocsparse_mat_descr)descrB,
                           nnzB,
                           (const rocsparse_float_complex*)csrSortedValB,
                           csrSortedRowPtrB,
                           csrSortedColIndB,
                           (const rocsparse_mat_descr)descrC,
                           (rocsparse_float_complex*)csrSortedValC,
                           csrSortedRowPtrC,
                           csrSortedColIndC));
}

hipsparseStatus_t hipsparseZcsrgeam2(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       n,
                                     const hipDoubleComplex*   alpha,
                                     const hipsparseMatDescr_t descrA,
                                     int                       nnzA,
                                     const hipDoubleComplex*   csrSortedValA,
                                     const int*                csrSortedRowPtrA,
                                     const int*                csrSortedColIndA,
                                     const hipDoubleComplex*   beta,
                                     const hipsparseMatDescr_t descrB,
                                     int                       nnzB,
                                     const hipDoubleComplex*   csrSortedValB,
                                     const int*                csrSortedRowPtrB,
                                     const int*                csrSortedColIndB,
                                     const hipsparseMatDescr_t descrC,
                                     hipDoubleComplex*         csrSortedValC,
                                     int*                      csrSortedRowPtrC,
                                     int*                      csrSortedColIndC,
                                     void*                     pBuffer)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_zcsrgeam((rocsparse_handle)handle,
                           m,
                           n,
                           (const rocsparse_double_complex*)alpha,
                           (const rocsparse_mat_descr)descrA,
                           nnzA,
                           (const rocsparse_double_complex*)csrSortedValA,
                           csrSortedRowPtrA,
                           csrSortedColIndA,
                           (const rocsparse_double_complex*)beta,
                           (const rocsparse_mat_descr)descrB,
                           nnzB,
                           (const rocsparse_double_complex*)csrSortedValB,
                           csrSortedRowPtrB,
                           csrSortedColIndB,
                           (const rocsparse_mat_descr)descrC,
                           (rocsparse_double_complex*)csrSortedValC,
                           csrSortedRowPtrC,
                           csrSortedColIndC));
}

hipsparseStatus_t hipsparseXcsrgemmNnz(hipsparseHandle_t         handle,
                                       hipsparseOperation_t      transA,
                                       hipsparseOperation_t      transB,
                                       int                       m,
                                       int                       n,
                                       int                       k,
                                       const hipsparseMatDescr_t descrA,
                                       int                       nnzA,
                                       const int*                csrRowPtrA,
                                       const int*                csrColIndA,
                                       const hipsparseMatDescr_t descrB,
                                       int                       nnzB,
                                       const int*                csrRowPtrB,
                                       const int*                csrColIndB,
                                       const hipsparseMatDescr_t descrC,
                                       int*                      csrRowPtrC,
                                       int*                      nnzTotalDevHostPtr)
{
    // Create matrix info
    rocsparse_mat_info info;
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_create_mat_info(&info));

    // Buffer
    size_t buffer_size;
    void*  temp_buffer;

    // Initialize alpha = 1.0
    hipDoubleComplex  one = make_hipDoubleComplex(1.0, 0.0);
    hipDoubleComplex* alpha;

    // Get pointer mode
    rocsparse_pointer_mode pointer_mode;
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_get_pointer_mode((rocsparse_handle)handle, &pointer_mode));

    if(pointer_mode == rocsparse_pointer_mode_host)
    {
        alpha  = (hipDoubleComplex*)malloc(sizeof(hipDoubleComplex));
        *alpha = one;
    }
    else
    {
        RETURN_IF_HIP_ERROR(hipMalloc((void**)&alpha, sizeof(hipDoubleComplex)));
        RETURN_IF_HIP_ERROR(
            hipMemcpy(alpha, &one, sizeof(hipDoubleComplex), hipMemcpyHostToDevice));
    }

    // Obtain temporary buffer size
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_zcsrgemm_buffer_size((rocsparse_handle)handle,
                                                             hipOperationToHCCOperation(transA),
                                                             hipOperationToHCCOperation(transB),
                                                             m,
                                                             n,
                                                             k,
                                                             (const rocsparse_double_complex*)alpha,
                                                             (rocsparse_mat_descr)descrA,
                                                             nnzA,
                                                             csrRowPtrA,
                                                             csrColIndA,
                                                             (rocsparse_mat_descr)descrB,
                                                             nnzB,
                                                             csrRowPtrB,
                                                             csrColIndB,
                                                             nullptr,
                                                             nullptr,
                                                             0,
                                                             nullptr,
                                                             nullptr,
                                                             info,
                                                             &buffer_size));

    RETURN_IF_HIP_ERROR(hipMalloc(&temp_buffer, buffer_size));

    // Determine nnz
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_csrgemm_nnz((rocsparse_handle)handle,
                                                    hipOperationToHCCOperation(transA),
                                                    hipOperationToHCCOperation(transB),
                                                    m,
                                                    n,
                                                    k,
                                                    (rocsparse_mat_descr)descrA,
                                                    nnzA,
                                                    csrRowPtrA,
                                                    csrColIndA,
                                                    (rocsparse_mat_descr)descrB,
                                                    nnzB,
                                                    csrRowPtrB,
                                                    csrColIndB,
                                                    nullptr,
                                                    0,
                                                    nullptr,
                                                    nullptr,
                                                    (rocsparse_mat_descr)descrC,
                                                    csrRowPtrC,
                                                    nnzTotalDevHostPtr,
                                                    info,
                                                    temp_buffer));

    if(pointer_mode == rocsparse_pointer_mode_host)
    {
        free(alpha);
    }
    else
    {
        RETURN_IF_HIP_ERROR(hipFree(alpha));
    }

    RETURN_IF_HIP_ERROR(hipFree(temp_buffer));

    RETURN_IF_ROCSPARSE_ERROR(rocsparse_destroy_mat_info(info));

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseScsrgemm(hipsparseHandle_t         handle,
                                    hipsparseOperation_t      transA,
                                    hipsparseOperation_t      transB,
                                    int                       m,
                                    int                       n,
                                    int                       k,
                                    const hipsparseMatDescr_t descrA,
                                    int                       nnzA,
                                    const float*              csrValA,
                                    const int*                csrRowPtrA,
                                    const int*                csrColIndA,
                                    const hipsparseMatDescr_t descrB,
                                    int                       nnzB,
                                    const float*              csrValB,
                                    const int*                csrRowPtrB,
                                    const int*                csrColIndB,
                                    const hipsparseMatDescr_t descrC,
                                    float*                    csrValC,
                                    const int*                csrRowPtrC,
                                    int*                      csrColIndC)
{
    // Create matrix info
    rocsparse_mat_info info;
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_create_mat_info(&info));

    // Buffer
    size_t buffer_size;
    void*  temp_buffer;

    // Initialize alpha = 1.0
    float  one = 1.0f;
    float* alpha;

    // Get pointer mode
    rocsparse_pointer_mode pointer_mode;
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_get_pointer_mode((rocsparse_handle)handle, &pointer_mode));

    if(pointer_mode == rocsparse_pointer_mode_host)
    {
        alpha  = (float*)malloc(sizeof(float));
        *alpha = one;
    }
    else
    {
        RETURN_IF_HIP_ERROR(hipMalloc((void**)&alpha, sizeof(float)));
        RETURN_IF_HIP_ERROR(hipMemcpy(alpha, &one, sizeof(float), hipMemcpyHostToDevice));
    }

    // Obtain temporary buffer size
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_scsrgemm_buffer_size((rocsparse_handle)handle,
                                                             hipOperationToHCCOperation(transA),
                                                             hipOperationToHCCOperation(transB),
                                                             m,
                                                             n,
                                                             k,
                                                             alpha,
                                                             (rocsparse_mat_descr)descrA,
                                                             nnzA,
                                                             csrRowPtrA,
                                                             csrColIndA,
                                                             (rocsparse_mat_descr)descrB,
                                                             nnzB,
                                                             csrRowPtrB,
                                                             csrColIndB,
                                                             nullptr,
                                                             nullptr,
                                                             0,
                                                             nullptr,
                                                             nullptr,
                                                             info,
                                                             &buffer_size));

    RETURN_IF_HIP_ERROR(hipMalloc(&temp_buffer, buffer_size));

    // Perform csrgemm computation
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_scsrgemm((rocsparse_handle)handle,
                                                 hipOperationToHCCOperation(transA),
                                                 hipOperationToHCCOperation(transB),
                                                 m,
                                                 n,
                                                 k,
                                                 alpha,
                                                 (rocsparse_mat_descr)descrA,
                                                 nnzA,
                                                 csrValA,
                                                 csrRowPtrA,
                                                 csrColIndA,
                                                 (rocsparse_mat_descr)descrB,
                                                 nnzB,
                                                 csrValB,
                                                 csrRowPtrB,
                                                 csrColIndB,
                                                 nullptr,
                                                 nullptr,
                                                 0,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 (rocsparse_mat_descr)descrC,
                                                 csrValC,
                                                 csrRowPtrC,
                                                 csrColIndC,
                                                 info,
                                                 temp_buffer));

    if(pointer_mode == rocsparse_pointer_mode_host)
    {
        free(alpha);
    }
    else
    {
        RETURN_IF_HIP_ERROR(hipFree(alpha));
    }

    RETURN_IF_HIP_ERROR(hipFree(temp_buffer));

    RETURN_IF_ROCSPARSE_ERROR(rocsparse_destroy_mat_info(info));

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseDcsrgemm(hipsparseHandle_t         handle,
                                    hipsparseOperation_t      transA,
                                    hipsparseOperation_t      transB,
                                    int                       m,
                                    int                       n,
                                    int                       k,
                                    const hipsparseMatDescr_t descrA,
                                    int                       nnzA,
                                    const double*             csrValA,
                                    const int*                csrRowPtrA,
                                    const int*                csrColIndA,
                                    const hipsparseMatDescr_t descrB,
                                    int                       nnzB,
                                    const double*             csrValB,
                                    const int*                csrRowPtrB,
                                    const int*                csrColIndB,
                                    const hipsparseMatDescr_t descrC,
                                    double*                   csrValC,
                                    const int*                csrRowPtrC,
                                    int*                      csrColIndC)
{
    // Create matrix info
    rocsparse_mat_info info;
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_create_mat_info(&info));

    // Buffer
    size_t buffer_size;
    void*  temp_buffer;

    // Initialize alpha = 1.0
    double  one = 1.0;
    double* alpha;

    // Get pointer mode
    rocsparse_pointer_mode pointer_mode;
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_get_pointer_mode((rocsparse_handle)handle, &pointer_mode));

    if(pointer_mode == rocsparse_pointer_mode_host)
    {
        alpha  = (double*)malloc(sizeof(double));
        *alpha = one;
    }
    else
    {
        RETURN_IF_HIP_ERROR(hipMalloc((void**)&alpha, sizeof(double)));
        RETURN_IF_HIP_ERROR(hipMemcpy(alpha, &one, sizeof(double), hipMemcpyHostToDevice));
    }

    // Obtain temporary buffer size
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_dcsrgemm_buffer_size((rocsparse_handle)handle,
                                                             hipOperationToHCCOperation(transA),
                                                             hipOperationToHCCOperation(transB),
                                                             m,
                                                             n,
                                                             k,
                                                             alpha,
                                                             (rocsparse_mat_descr)descrA,
                                                             nnzA,
                                                             csrRowPtrA,
                                                             csrColIndA,
                                                             (rocsparse_mat_descr)descrB,
                                                             nnzB,
                                                             csrRowPtrB,
                                                             csrColIndB,
                                                             nullptr,
                                                             nullptr,
                                                             0,
                                                             nullptr,
                                                             nullptr,
                                                             info,
                                                             &buffer_size));

    RETURN_IF_HIP_ERROR(hipMalloc(&temp_buffer, buffer_size));

    // Perform csrgemm computation
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_dcsrgemm((rocsparse_handle)handle,
                                                 hipOperationToHCCOperation(transA),
                                                 hipOperationToHCCOperation(transB),
                                                 m,
                                                 n,
                                                 k,
                                                 alpha,
                                                 (rocsparse_mat_descr)descrA,
                                                 nnzA,
                                                 csrValA,
                                                 csrRowPtrA,
                                                 csrColIndA,
                                                 (rocsparse_mat_descr)descrB,
                                                 nnzB,
                                                 csrValB,
                                                 csrRowPtrB,
                                                 csrColIndB,
                                                 nullptr,
                                                 nullptr,
                                                 0,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 (rocsparse_mat_descr)descrC,
                                                 csrValC,
                                                 csrRowPtrC,
                                                 csrColIndC,
                                                 info,
                                                 temp_buffer));

    if(pointer_mode == rocsparse_pointer_mode_host)
    {
        free(alpha);
    }
    else
    {
        RETURN_IF_HIP_ERROR(hipFree(alpha));
    }

    RETURN_IF_HIP_ERROR(hipFree(temp_buffer));

    RETURN_IF_ROCSPARSE_ERROR(rocsparse_destroy_mat_info(info));

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseCcsrgemm(hipsparseHandle_t         handle,
                                    hipsparseOperation_t      transA,
                                    hipsparseOperation_t      transB,
                                    int                       m,
                                    int                       n,
                                    int                       k,
                                    const hipsparseMatDescr_t descrA,
                                    int                       nnzA,
                                    const hipComplex*         csrValA,
                                    const int*                csrRowPtrA,
                                    const int*                csrColIndA,
                                    const hipsparseMatDescr_t descrB,
                                    int                       nnzB,
                                    const hipComplex*         csrValB,
                                    const int*                csrRowPtrB,
                                    const int*                csrColIndB,
                                    const hipsparseMatDescr_t descrC,
                                    hipComplex*               csrValC,
                                    const int*                csrRowPtrC,
                                    int*                      csrColIndC)
{
    // Create matrix info
    rocsparse_mat_info info;
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_create_mat_info(&info));

    // Buffer
    size_t buffer_size;
    void*  temp_buffer;

    // Initialize alpha = 1.0
    hipComplex  one = make_hipComplex(1.0f, 0.0f);
    hipComplex* alpha;

    // Get pointer mode
    rocsparse_pointer_mode pointer_mode;
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_get_pointer_mode((rocsparse_handle)handle, &pointer_mode));

    if(pointer_mode == rocsparse_pointer_mode_host)
    {
        alpha  = (hipComplex*)malloc(sizeof(hipComplex));
        *alpha = one;
    }
    else
    {
        RETURN_IF_HIP_ERROR(hipMalloc((void**)&alpha, sizeof(hipComplex)));
        RETURN_IF_HIP_ERROR(hipMemcpy(alpha, &one, sizeof(hipComplex), hipMemcpyHostToDevice));
    }

    // Obtain temporary buffer size
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_ccsrgemm_buffer_size((rocsparse_handle)handle,
                                                             hipOperationToHCCOperation(transA),
                                                             hipOperationToHCCOperation(transB),
                                                             m,
                                                             n,
                                                             k,
                                                             (const rocsparse_float_complex*)alpha,
                                                             (rocsparse_mat_descr)descrA,
                                                             nnzA,
                                                             csrRowPtrA,
                                                             csrColIndA,
                                                             (rocsparse_mat_descr)descrB,
                                                             nnzB,
                                                             csrRowPtrB,
                                                             csrColIndB,
                                                             nullptr,
                                                             nullptr,
                                                             0,
                                                             nullptr,
                                                             nullptr,
                                                             info,
                                                             &buffer_size));

    RETURN_IF_HIP_ERROR(hipMalloc(&temp_buffer, buffer_size));

    // Perform csrgemm computation
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_ccsrgemm((rocsparse_handle)handle,
                                                 hipOperationToHCCOperation(transA),
                                                 hipOperationToHCCOperation(transB),
                                                 m,
                                                 n,
                                                 k,
                                                 (const rocsparse_float_complex*)alpha,
                                                 (rocsparse_mat_descr)descrA,
                                                 nnzA,
                                                 (const rocsparse_float_complex*)csrValA,
                                                 csrRowPtrA,
                                                 csrColIndA,
                                                 (rocsparse_mat_descr)descrB,
                                                 nnzB,
                                                 (const rocsparse_float_complex*)csrValB,
                                                 csrRowPtrB,
                                                 csrColIndB,
                                                 nullptr,
                                                 nullptr,
                                                 0,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 (rocsparse_mat_descr)descrC,
                                                 (rocsparse_float_complex*)csrValC,
                                                 csrRowPtrC,
                                                 csrColIndC,
                                                 info,
                                                 temp_buffer));

    if(pointer_mode == rocsparse_pointer_mode_host)
    {
        free(alpha);
    }
    else
    {
        RETURN_IF_HIP_ERROR(hipFree(alpha));
    }

    RETURN_IF_HIP_ERROR(hipFree(temp_buffer));

    RETURN_IF_ROCSPARSE_ERROR(rocsparse_destroy_mat_info(info));

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseZcsrgemm(hipsparseHandle_t         handle,
                                    hipsparseOperation_t      transA,
                                    hipsparseOperation_t      transB,
                                    int                       m,
                                    int                       n,
                                    int                       k,
                                    const hipsparseMatDescr_t descrA,
                                    int                       nnzA,
                                    const hipDoubleComplex*   csrValA,
                                    const int*                csrRowPtrA,
                                    const int*                csrColIndA,
                                    const hipsparseMatDescr_t descrB,
                                    int                       nnzB,
                                    const hipDoubleComplex*   csrValB,
                                    const int*                csrRowPtrB,
                                    const int*                csrColIndB,
                                    const hipsparseMatDescr_t descrC,
                                    hipDoubleComplex*         csrValC,
                                    const int*                csrRowPtrC,
                                    int*                      csrColIndC)
{
    // Create matrix info
    rocsparse_mat_info info;
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_create_mat_info(&info));

    // Buffer
    size_t buffer_size;
    void*  temp_buffer;

    // Initialize alpha = 1.0
    hipDoubleComplex  one = make_hipDoubleComplex(1.0, 0.0);
    hipDoubleComplex* alpha;

    // Get pointer mode
    rocsparse_pointer_mode pointer_mode;
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_get_pointer_mode((rocsparse_handle)handle, &pointer_mode));

    if(pointer_mode == rocsparse_pointer_mode_host)
    {
        alpha  = (hipDoubleComplex*)malloc(sizeof(hipDoubleComplex));
        *alpha = one;
    }
    else
    {
        RETURN_IF_HIP_ERROR(hipMalloc((void**)&alpha, sizeof(hipDoubleComplex)));
        RETURN_IF_HIP_ERROR(
            hipMemcpy(alpha, &one, sizeof(hipDoubleComplex), hipMemcpyHostToDevice));
    }

    // Obtain temporary buffer size
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_zcsrgemm_buffer_size((rocsparse_handle)handle,
                                                             hipOperationToHCCOperation(transA),
                                                             hipOperationToHCCOperation(transB),
                                                             m,
                                                             n,
                                                             k,
                                                             (const rocsparse_double_complex*)alpha,
                                                             (rocsparse_mat_descr)descrA,
                                                             nnzA,
                                                             csrRowPtrA,
                                                             csrColIndA,
                                                             (rocsparse_mat_descr)descrB,
                                                             nnzB,
                                                             csrRowPtrB,
                                                             csrColIndB,
                                                             nullptr,
                                                             nullptr,
                                                             0,
                                                             nullptr,
                                                             nullptr,
                                                             info,
                                                             &buffer_size));

    RETURN_IF_HIP_ERROR(hipMalloc(&temp_buffer, buffer_size));

    // Perform csrgemm computation
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_zcsrgemm((rocsparse_handle)handle,
                                                 hipOperationToHCCOperation(transA),
                                                 hipOperationToHCCOperation(transB),
                                                 m,
                                                 n,
                                                 k,
                                                 (const rocsparse_double_complex*)alpha,
                                                 (rocsparse_mat_descr)descrA,
                                                 nnzA,
                                                 (const rocsparse_double_complex*)csrValA,
                                                 csrRowPtrA,
                                                 csrColIndA,
                                                 (rocsparse_mat_descr)descrB,
                                                 nnzB,
                                                 (const rocsparse_double_complex*)csrValB,
                                                 csrRowPtrB,
                                                 csrColIndB,
                                                 nullptr,
                                                 nullptr,
                                                 0,
                                                 nullptr,
                                                 nullptr,
                                                 nullptr,
                                                 (rocsparse_mat_descr)descrC,
                                                 (rocsparse_double_complex*)csrValC,
                                                 csrRowPtrC,
                                                 csrColIndC,
                                                 info,
                                                 temp_buffer));

    if(pointer_mode == rocsparse_pointer_mode_host)
    {
        free(alpha);
    }
    else
    {
        RETURN_IF_HIP_ERROR(hipFree(alpha));
    }

    RETURN_IF_HIP_ERROR(hipFree(temp_buffer));

    RETURN_IF_ROCSPARSE_ERROR(rocsparse_destroy_mat_info(info));

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseScsrgemm2_bufferSizeExt(hipsparseHandle_t         handle,
                                                   int                       m,
                                                   int                       n,
                                                   int                       k,
                                                   const float*              alpha,
                                                   const hipsparseMatDescr_t descrA,
                                                   int                       nnzA,
                                                   const int*                csrRowPtrA,
                                                   const int*                csrColIndA,
                                                   const hipsparseMatDescr_t descrB,
                                                   int                       nnzB,
                                                   const int*                csrRowPtrB,
                                                   const int*                csrColIndB,
                                                   const float*              beta,
                                                   const hipsparseMatDescr_t descrD,
                                                   int                       nnzD,
                                                   const int*                csrRowPtrD,
                                                   const int*                csrColIndD,
                                                   csrgemm2Info_t            info,
                                                   size_t*                   pBufferSizeInBytes)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_scsrgemm_buffer_size((rocsparse_handle)handle,
                                                                     rocsparse_operation_none,
                                                                     rocsparse_operation_none,
                                                                     m,
                                                                     n,
                                                                     k,
                                                                     alpha,
                                                                     (rocsparse_mat_descr)descrA,
                                                                     nnzA,
                                                                     csrRowPtrA,
                                                                     csrColIndA,
                                                                     (rocsparse_mat_descr)descrB,
                                                                     nnzB,
                                                                     csrRowPtrB,
                                                                     csrColIndB,
                                                                     beta,
                                                                     (rocsparse_mat_descr)descrD,
                                                                     nnzD,
                                                                     csrRowPtrD,
                                                                     csrColIndD,
                                                                     (rocsparse_mat_info)info,
                                                                     pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseDcsrgemm2_bufferSizeExt(hipsparseHandle_t         handle,
                                                   int                       m,
                                                   int                       n,
                                                   int                       k,
                                                   const double*             alpha,
                                                   const hipsparseMatDescr_t descrA,
                                                   int                       nnzA,
                                                   const int*                csrRowPtrA,
                                                   const int*                csrColIndA,
                                                   const hipsparseMatDescr_t descrB,
                                                   int                       nnzB,
                                                   const int*                csrRowPtrB,
                                                   const int*                csrColIndB,
                                                   const double*             beta,
                                                   const hipsparseMatDescr_t descrD,
                                                   int                       nnzD,
                                                   const int*                csrRowPtrD,
                                                   const int*                csrColIndD,
                                                   csrgemm2Info_t            info,
                                                   size_t*                   pBufferSizeInBytes)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_dcsrgemm_buffer_size((rocsparse_handle)handle,
                                                                     rocsparse_operation_none,
                                                                     rocsparse_operation_none,
                                                                     m,
                                                                     n,
                                                                     k,
                                                                     alpha,
                                                                     (rocsparse_mat_descr)descrA,
                                                                     nnzA,
                                                                     csrRowPtrA,
                                                                     csrColIndA,
                                                                     (rocsparse_mat_descr)descrB,
                                                                     nnzB,
                                                                     csrRowPtrB,
                                                                     csrColIndB,
                                                                     beta,
                                                                     (rocsparse_mat_descr)descrD,
                                                                     nnzD,
                                                                     csrRowPtrD,
                                                                     csrColIndD,
                                                                     (rocsparse_mat_info)info,
                                                                     pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseCcsrgemm2_bufferSizeExt(hipsparseHandle_t         handle,
                                                   int                       m,
                                                   int                       n,
                                                   int                       k,
                                                   const hipComplex*         alpha,
                                                   const hipsparseMatDescr_t descrA,
                                                   int                       nnzA,
                                                   const int*                csrRowPtrA,
                                                   const int*                csrColIndA,
                                                   const hipsparseMatDescr_t descrB,
                                                   int                       nnzB,
                                                   const int*                csrRowPtrB,
                                                   const int*                csrColIndB,
                                                   const hipComplex*         beta,
                                                   const hipsparseMatDescr_t descrD,
                                                   int                       nnzD,
                                                   const int*                csrRowPtrD,
                                                   const int*                csrColIndD,
                                                   csrgemm2Info_t            info,
                                                   size_t*                   pBufferSizeInBytes)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_ccsrgemm_buffer_size((rocsparse_handle)handle,
                                       rocsparse_operation_none,
                                       rocsparse_operation_none,
                                       m,
                                       n,
                                       k,
                                       (const rocsparse_float_complex*)alpha,
                                       (rocsparse_mat_descr)descrA,
                                       nnzA,
                                       csrRowPtrA,
                                       csrColIndA,
                                       (rocsparse_mat_descr)descrB,
                                       nnzB,
                                       csrRowPtrB,
                                       csrColIndB,
                                       (const rocsparse_float_complex*)beta,
                                       (rocsparse_mat_descr)descrD,
                                       nnzD,
                                       csrRowPtrD,
                                       csrColIndD,
                                       (rocsparse_mat_info)info,
                                       pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseZcsrgemm2_bufferSizeExt(hipsparseHandle_t         handle,
                                                   int                       m,
                                                   int                       n,
                                                   int                       k,
                                                   const hipDoubleComplex*   alpha,
                                                   const hipsparseMatDescr_t descrA,
                                                   int                       nnzA,
                                                   const int*                csrRowPtrA,
                                                   const int*                csrColIndA,
                                                   const hipsparseMatDescr_t descrB,
                                                   int                       nnzB,
                                                   const int*                csrRowPtrB,
                                                   const int*                csrColIndB,
                                                   const hipDoubleComplex*   beta,
                                                   const hipsparseMatDescr_t descrD,
                                                   int                       nnzD,
                                                   const int*                csrRowPtrD,
                                                   const int*                csrColIndD,
                                                   csrgemm2Info_t            info,
                                                   size_t*                   pBufferSizeInBytes)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_zcsrgemm_buffer_size((rocsparse_handle)handle,
                                       rocsparse_operation_none,
                                       rocsparse_operation_none,
                                       m,
                                       n,
                                       k,
                                       (const rocsparse_double_complex*)alpha,
                                       (rocsparse_mat_descr)descrA,
                                       nnzA,
                                       csrRowPtrA,
                                       csrColIndA,
                                       (rocsparse_mat_descr)descrB,
                                       nnzB,
                                       csrRowPtrB,
                                       csrColIndB,
                                       (const rocsparse_double_complex*)beta,
                                       (rocsparse_mat_descr)descrD,
                                       nnzD,
                                       csrRowPtrD,
                                       csrColIndD,
                                       (rocsparse_mat_info)info,
                                       pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseXcsrgemm2Nnz(hipsparseHandle_t         handle,
                                        int                       m,
                                        int                       n,
                                        int                       k,
                                        const hipsparseMatDescr_t descrA,
                                        int                       nnzA,
                                        const int*                csrRowPtrA,
                                        const int*                csrColIndA,
                                        const hipsparseMatDescr_t descrB,
                                        int                       nnzB,
                                        const int*                csrRowPtrB,
                                        const int*                csrColIndB,
                                        const hipsparseMatDescr_t descrD,
                                        int                       nnzD,
                                        const int*                csrRowPtrD,
                                        const int*                csrColIndD,
                                        const hipsparseMatDescr_t descrC,
                                        int*                      csrRowPtrC,
                                        int*                      nnzTotalDevHostPtr,
                                        const csrgemm2Info_t      info,
                                        void*                     pBuffer)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_csrgemm_nnz((rocsparse_handle)handle,
                                                            rocsparse_operation_none,
                                                            rocsparse_operation_none,
                                                            m,
                                                            n,
                                                            k,
                                                            (rocsparse_mat_descr)descrA,
                                                            nnzA,
                                                            csrRowPtrA,
                                                            csrColIndA,
                                                            (rocsparse_mat_descr)descrB,
                                                            nnzB,
                                                            csrRowPtrB,
                                                            csrColIndB,
                                                            (rocsparse_mat_descr)descrD,
                                                            nnzD,
                                                            csrRowPtrD,
                                                            csrColIndD,
                                                            (rocsparse_mat_descr)descrC,
                                                            csrRowPtrC,
                                                            nnzTotalDevHostPtr,
                                                            (rocsparse_mat_info)info,
                                                            pBuffer));
}

hipsparseStatus_t hipsparseScsrgemm2(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       n,
                                     int                       k,
                                     const float*              alpha,
                                     const hipsparseMatDescr_t descrA,
                                     int                       nnzA,
                                     const float*              csrValA,
                                     const int*                csrRowPtrA,
                                     const int*                csrColIndA,
                                     const hipsparseMatDescr_t descrB,
                                     int                       nnzB,
                                     const float*              csrValB,
                                     const int*                csrRowPtrB,
                                     const int*                csrColIndB,
                                     const float*              beta,
                                     const hipsparseMatDescr_t descrD,
                                     int                       nnzD,
                                     const float*              csrValD,
                                     const int*                csrRowPtrD,
                                     const int*                csrColIndD,
                                     const hipsparseMatDescr_t descrC,
                                     float*                    csrValC,
                                     const int*                csrRowPtrC,
                                     int*                      csrColIndC,
                                     const csrgemm2Info_t      info,
                                     void*                     pBuffer)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_scsrgemm((rocsparse_handle)handle,
                                                         rocsparse_operation_none,
                                                         rocsparse_operation_none,
                                                         m,
                                                         n,
                                                         k,
                                                         alpha,
                                                         (rocsparse_mat_descr)descrA,
                                                         nnzA,
                                                         csrValA,
                                                         csrRowPtrA,
                                                         csrColIndA,
                                                         (rocsparse_mat_descr)descrB,
                                                         nnzB,
                                                         csrValB,
                                                         csrRowPtrB,
                                                         csrColIndB,
                                                         beta,
                                                         (rocsparse_mat_descr)descrD,
                                                         nnzD,
                                                         csrValD,
                                                         csrRowPtrD,
                                                         csrColIndD,
                                                         (rocsparse_mat_descr)descrC,
                                                         csrValC,
                                                         csrRowPtrC,
                                                         csrColIndC,
                                                         (rocsparse_mat_info)info,
                                                         pBuffer));
}

hipsparseStatus_t hipsparseDcsrgemm2(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       n,
                                     int                       k,
                                     const double*             alpha,
                                     const hipsparseMatDescr_t descrA,
                                     int                       nnzA,
                                     const double*             csrValA,
                                     const int*                csrRowPtrA,
                                     const int*                csrColIndA,
                                     const hipsparseMatDescr_t descrB,
                                     int                       nnzB,
                                     const double*             csrValB,
                                     const int*                csrRowPtrB,
                                     const int*                csrColIndB,
                                     const double*             beta,
                                     const hipsparseMatDescr_t descrD,
                                     int                       nnzD,
                                     const double*             csrValD,
                                     const int*                csrRowPtrD,
                                     const int*                csrColIndD,
                                     const hipsparseMatDescr_t descrC,
                                     double*                   csrValC,
                                     const int*                csrRowPtrC,
                                     int*                      csrColIndC,
                                     const csrgemm2Info_t      info,
                                     void*                     pBuffer)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_dcsrgemm((rocsparse_handle)handle,
                                                         rocsparse_operation_none,
                                                         rocsparse_operation_none,
                                                         m,
                                                         n,
                                                         k,
                                                         alpha,
                                                         (rocsparse_mat_descr)descrA,
                                                         nnzA,
                                                         csrValA,
                                                         csrRowPtrA,
                                                         csrColIndA,
                                                         (rocsparse_mat_descr)descrB,
                                                         nnzB,
                                                         csrValB,
                                                         csrRowPtrB,
                                                         csrColIndB,
                                                         beta,
                                                         (rocsparse_mat_descr)descrD,
                                                         nnzD,
                                                         csrValD,
                                                         csrRowPtrD,
                                                         csrColIndD,
                                                         (rocsparse_mat_descr)descrC,
                                                         csrValC,
                                                         csrRowPtrC,
                                                         csrColIndC,
                                                         (rocsparse_mat_info)info,
                                                         pBuffer));
}

hipsparseStatus_t hipsparseCcsrgemm2(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       n,
                                     int                       k,
                                     const hipComplex*         alpha,
                                     const hipsparseMatDescr_t descrA,
                                     int                       nnzA,
                                     const hipComplex*         csrValA,
                                     const int*                csrRowPtrA,
                                     const int*                csrColIndA,
                                     const hipsparseMatDescr_t descrB,
                                     int                       nnzB,
                                     const hipComplex*         csrValB,
                                     const int*                csrRowPtrB,
                                     const int*                csrColIndB,
                                     const hipComplex*         beta,
                                     const hipsparseMatDescr_t descrD,
                                     int                       nnzD,
                                     const hipComplex*         csrValD,
                                     const int*                csrRowPtrD,
                                     const int*                csrColIndD,
                                     const hipsparseMatDescr_t descrC,
                                     hipComplex*               csrValC,
                                     const int*                csrRowPtrC,
                                     int*                      csrColIndC,
                                     const csrgemm2Info_t      info,
                                     void*                     pBuffer)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_ccsrgemm((rocsparse_handle)handle,
                                                         rocsparse_operation_none,
                                                         rocsparse_operation_none,
                                                         m,
                                                         n,
                                                         k,
                                                         (const rocsparse_float_complex*)alpha,
                                                         (rocsparse_mat_descr)descrA,
                                                         nnzA,
                                                         (const rocsparse_float_complex*)csrValA,
                                                         csrRowPtrA,
                                                         csrColIndA,
                                                         (rocsparse_mat_descr)descrB,
                                                         nnzB,
                                                         (const rocsparse_float_complex*)csrValB,
                                                         csrRowPtrB,
                                                         csrColIndB,
                                                         (const rocsparse_float_complex*)beta,
                                                         (rocsparse_mat_descr)descrD,
                                                         nnzD,
                                                         (const rocsparse_float_complex*)csrValD,
                                                         csrRowPtrD,
                                                         csrColIndD,
                                                         (rocsparse_mat_descr)descrC,
                                                         (rocsparse_float_complex*)csrValC,
                                                         csrRowPtrC,
                                                         csrColIndC,
                                                         (rocsparse_mat_info)info,
                                                         pBuffer));
}

hipsparseStatus_t hipsparseZcsrgemm2(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       n,
                                     int                       k,
                                     const hipDoubleComplex*   alpha,
                                     const hipsparseMatDescr_t descrA,
                                     int                       nnzA,
                                     const hipDoubleComplex*   csrValA,
                                     const int*                csrRowPtrA,
                                     const int*                csrColIndA,
                                     const hipsparseMatDescr_t descrB,
                                     int                       nnzB,
                                     const hipDoubleComplex*   csrValB,
                                     const int*                csrRowPtrB,
                                     const int*                csrColIndB,
                                     const hipDoubleComplex*   beta,
                                     const hipsparseMatDescr_t descrD,
                                     int                       nnzD,
                                     const hipDoubleComplex*   csrValD,
                                     const int*                csrRowPtrD,
                                     const int*                csrColIndD,
                                     const hipsparseMatDescr_t descrC,
                                     hipDoubleComplex*         csrValC,
                                     const int*                csrRowPtrC,
                                     int*                      csrColIndC,
                                     const csrgemm2Info_t      info,
                                     void*                     pBuffer)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_zcsrgemm((rocsparse_handle)handle,
                                                         rocsparse_operation_none,
                                                         rocsparse_operation_none,
                                                         m,
                                                         n,
                                                         k,
                                                         (const rocsparse_double_complex*)alpha,
                                                         (rocsparse_mat_descr)descrA,
                                                         nnzA,
                                                         (const rocsparse_double_complex*)csrValA,
                                                         csrRowPtrA,
                                                         csrColIndA,
                                                         (rocsparse_mat_descr)descrB,
                                                         nnzB,
                                                         (const rocsparse_double_complex*)csrValB,
                                                         csrRowPtrB,
                                                         csrColIndB,
                                                         (const rocsparse_double_complex*)beta,
                                                         (rocsparse_mat_descr)descrD,
                                                         nnzD,
                                                         (const rocsparse_double_complex*)csrValD,
                                                         csrRowPtrD,
                                                         csrColIndD,
                                                         (rocsparse_mat_descr)descrC,
                                                         (rocsparse_double_complex*)csrValC,
                                                         csrRowPtrC,
                                                         csrColIndC,
                                                         (rocsparse_mat_info)info,
                                                         pBuffer));
}

hipsparseStatus_t
    hipsparseXbsrilu02_zeroPivot(hipsparseHandle_t handle, bsrilu02Info_t info, int* position)
{
    // Obtain stream, to explicitly sync (cusparse bsrilu02_zeropivot is blocking)
    hipStream_t stream;
    RETURN_IF_HIPSPARSE_ERROR(hipsparseGetStream(handle, &stream));

    // bsrilu0 zero pivot
    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse_bsrilu0_zero_pivot((rocsparse_handle)handle, (rocsparse_mat_info)info, position));

    // Synchronize stream
    RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseSbsrilu02_numericBoost(
    hipsparseHandle_t handle, bsrilu02Info_t info, int enable_boost, double* tol, float* boost_val)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_dsbsrilu0_numeric_boost(
        (rocsparse_handle)handle, (rocsparse_mat_info)info, enable_boost, tol, boost_val));
}

hipsparseStatus_t hipsparseDbsrilu02_numericBoost(
    hipsparseHandle_t handle, bsrilu02Info_t info, int enable_boost, double* tol, double* boost_val)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_dbsrilu0_numeric_boost(
        (rocsparse_handle)handle, (rocsparse_mat_info)info, enable_boost, tol, boost_val));
}

hipsparseStatus_t hipsparseCbsrilu02_numericBoost(hipsparseHandle_t handle,
                                                  bsrilu02Info_t    info,
                                                  int               enable_boost,
                                                  double*           tol,
                                                  hipComplex*       boost_val)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_dcbsrilu0_numeric_boost((rocsparse_handle)handle,
                                          (rocsparse_mat_info)info,
                                          enable_boost,
                                          tol,
                                          (rocsparse_float_complex*)boost_val));
}

hipsparseStatus_t hipsparseZbsrilu02_numericBoost(hipsparseHandle_t handle,
                                                  bsrilu02Info_t    info,
                                                  int               enable_boost,
                                                  double*           tol,
                                                  hipDoubleComplex* boost_val)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_zbsrilu0_numeric_boost((rocsparse_handle)handle,
                                         (rocsparse_mat_info)info,
                                         enable_boost,
                                         tol,
                                         (rocsparse_double_complex*)boost_val));
}

hipsparseStatus_t hipsparseSbsrilu02_bufferSize(hipsparseHandle_t         handle,
                                                hipsparseDirection_t      dirA,
                                                int                       mb,
                                                int                       nnzb,
                                                const hipsparseMatDescr_t descrA,
                                                float*                    bsrValA,
                                                const int*                bsrRowPtrA,
                                                const int*                bsrColIndA,
                                                int                       blockDim,
                                                bsrilu02Info_t            info,
                                                int*                      pBufferSizeInBytes)
{
    if(pBufferSizeInBytes == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    size_t           buffer_size;
    rocsparse_status status;

    status = rocsparse_sbsrilu0_buffer_size((rocsparse_handle)handle,
                                            hipDirectionToHCCDirection(dirA),
                                            mb,
                                            nnzb,
                                            (rocsparse_mat_descr)descrA,
                                            bsrValA,
                                            bsrRowPtrA,
                                            bsrColIndA,
                                            blockDim,
                                            (rocsparse_mat_info)info,
                                            &buffer_size);

    *pBufferSizeInBytes = (int)buffer_size;

    return rocSPARSEStatusToHIPStatus(status);
}

hipsparseStatus_t hipsparseDbsrilu02_bufferSize(hipsparseHandle_t         handle,
                                                hipsparseDirection_t      dirA,
                                                int                       mb,
                                                int                       nnzb,
                                                const hipsparseMatDescr_t descrA,
                                                double*                   bsrValA,
                                                const int*                bsrRowPtrA,
                                                const int*                bsrColIndA,
                                                int                       blockDim,
                                                bsrilu02Info_t            info,
                                                int*                      pBufferSizeInBytes)
{
    if(pBufferSizeInBytes == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    size_t           buffer_size;
    rocsparse_status status;

    status = rocsparse_dbsrilu0_buffer_size((rocsparse_handle)handle,
                                            hipDirectionToHCCDirection(dirA),
                                            mb,
                                            nnzb,
                                            (rocsparse_mat_descr)descrA,
                                            bsrValA,
                                            bsrRowPtrA,
                                            bsrColIndA,
                                            blockDim,
                                            (rocsparse_mat_info)info,
                                            &buffer_size);

    *pBufferSizeInBytes = (int)buffer_size;

    return rocSPARSEStatusToHIPStatus(status);
}

hipsparseStatus_t hipsparseCbsrilu02_bufferSize(hipsparseHandle_t         handle,
                                                hipsparseDirection_t      dirA,
                                                int                       mb,
                                                int                       nnzb,
                                                const hipsparseMatDescr_t descrA,
                                                hipComplex*               bsrValA,
                                                const int*                bsrRowPtrA,
                                                const int*                bsrColIndA,
                                                int                       blockDim,
                                                bsrilu02Info_t            info,
                                                int*                      pBufferSizeInBytes)
{
    if(pBufferSizeInBytes == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    size_t           buffer_size;
    rocsparse_status status;

    status = rocsparse_cbsrilu0_buffer_size((rocsparse_handle)handle,
                                            hipDirectionToHCCDirection(dirA),
                                            mb,
                                            nnzb,
                                            (rocsparse_mat_descr)descrA,
                                            (rocsparse_float_complex*)bsrValA,
                                            bsrRowPtrA,
                                            bsrColIndA,
                                            blockDim,
                                            (rocsparse_mat_info)info,
                                            &buffer_size);

    *pBufferSizeInBytes = (int)buffer_size;

    return rocSPARSEStatusToHIPStatus(status);
}

hipsparseStatus_t hipsparseZbsrilu02_bufferSize(hipsparseHandle_t         handle,
                                                hipsparseDirection_t      dirA,
                                                int                       mb,
                                                int                       nnzb,
                                                const hipsparseMatDescr_t descrA,
                                                hipDoubleComplex*         bsrValA,
                                                const int*                bsrRowPtrA,
                                                const int*                bsrColIndA,
                                                int                       blockDim,
                                                bsrilu02Info_t            info,
                                                int*                      pBufferSizeInBytes)
{
    if(pBufferSizeInBytes == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    size_t           buffer_size;
    rocsparse_status status;

    status = rocsparse_zbsrilu0_buffer_size((rocsparse_handle)handle,
                                            hipDirectionToHCCDirection(dirA),
                                            mb,
                                            nnzb,
                                            (rocsparse_mat_descr)descrA,
                                            (rocsparse_double_complex*)bsrValA,
                                            bsrRowPtrA,
                                            bsrColIndA,
                                            blockDim,
                                            (rocsparse_mat_info)info,
                                            &buffer_size);

    *pBufferSizeInBytes = (int)buffer_size;

    return rocSPARSEStatusToHIPStatus(status);
}

hipsparseStatus_t hipsparseSbsrilu02_analysis(hipsparseHandle_t         handle,
                                              hipsparseDirection_t      dirA,
                                              int                       mb,
                                              int                       nnzb,
                                              const hipsparseMatDescr_t descrA,
                                              float*                    bsrValA,
                                              const int*                bsrRowPtrA,
                                              const int*                bsrColIndA,
                                              int                       blockDim,
                                              bsrilu02Info_t            info,
                                              hipsparseSolvePolicy_t    policy,
                                              void*                     pBuffer)
{
    // Obtain stream, to explicitly sync (cusparse bsrilu02_analysis is blocking)
    hipStream_t stream;
    RETURN_IF_HIPSPARSE_ERROR(hipsparseGetStream(handle, &stream));

    // bsrilu0 analysis
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_sbsrilu0_analysis((rocsparse_handle)handle,
                                                          hipDirectionToHCCDirection(dirA),
                                                          mb,
                                                          nnzb,
                                                          (rocsparse_mat_descr)descrA,
                                                          bsrValA,
                                                          bsrRowPtrA,
                                                          bsrColIndA,
                                                          blockDim,
                                                          (rocsparse_mat_info)info,
                                                          rocsparse_analysis_policy_force,
                                                          rocsparse_solve_policy_auto,
                                                          pBuffer));

    // Synchronize stream
    RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseDbsrilu02_analysis(hipsparseHandle_t         handle,
                                              hipsparseDirection_t      dirA,
                                              int                       mb,
                                              int                       nnzb,
                                              const hipsparseMatDescr_t descrA,
                                              double*                   bsrValA,
                                              const int*                bsrRowPtrA,
                                              const int*                bsrColIndA,
                                              int                       blockDim,
                                              bsrilu02Info_t            info,
                                              hipsparseSolvePolicy_t    policy,
                                              void*                     pBuffer)
{
    // Obtain stream, to explicitly sync (cusparse bsrilu02_analysis is blocking)
    hipStream_t stream;
    RETURN_IF_HIPSPARSE_ERROR(hipsparseGetStream(handle, &stream));

    // bsrilu0 analysis
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_dbsrilu0_analysis((rocsparse_handle)handle,
                                                          hipDirectionToHCCDirection(dirA),
                                                          mb,
                                                          nnzb,
                                                          (rocsparse_mat_descr)descrA,
                                                          bsrValA,
                                                          bsrRowPtrA,
                                                          bsrColIndA,
                                                          blockDim,
                                                          (rocsparse_mat_info)info,
                                                          rocsparse_analysis_policy_force,
                                                          rocsparse_solve_policy_auto,
                                                          pBuffer));

    // Synchronize stream
    RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseCbsrilu02_analysis(hipsparseHandle_t         handle,
                                              hipsparseDirection_t      dirA,
                                              int                       mb,
                                              int                       nnzb,
                                              const hipsparseMatDescr_t descrA,
                                              hipComplex*               bsrValA,
                                              const int*                bsrRowPtrA,
                                              const int*                bsrColIndA,
                                              int                       blockDim,
                                              bsrilu02Info_t            info,
                                              hipsparseSolvePolicy_t    policy,
                                              void*                     pBuffer)
{
    // Obtain stream, to explicitly sync (cusparse bsrilu02_analysis is blocking)
    hipStream_t stream;
    RETURN_IF_HIPSPARSE_ERROR(hipsparseGetStream(handle, &stream));

    // bsrilu0 analysis
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_cbsrilu0_analysis((rocsparse_handle)handle,
                                                          hipDirectionToHCCDirection(dirA),
                                                          mb,
                                                          nnzb,
                                                          (rocsparse_mat_descr)descrA,
                                                          (const rocsparse_float_complex*)bsrValA,
                                                          bsrRowPtrA,
                                                          bsrColIndA,
                                                          blockDim,
                                                          (rocsparse_mat_info)info,
                                                          rocsparse_analysis_policy_force,
                                                          rocsparse_solve_policy_auto,
                                                          pBuffer));

    // Synchronize stream
    RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseZbsrilu02_analysis(hipsparseHandle_t         handle,
                                              hipsparseDirection_t      dirA,
                                              int                       mb,
                                              int                       nnzb,
                                              const hipsparseMatDescr_t descrA,
                                              hipDoubleComplex*         bsrValA,
                                              const int*                bsrRowPtrA,
                                              const int*                bsrColIndA,
                                              int                       blockDim,
                                              bsrilu02Info_t            info,
                                              hipsparseSolvePolicy_t    policy,
                                              void*                     pBuffer)
{
    // Obtain stream, to explicitly sync (cusparse bsrilu02_analysis is blocking)
    hipStream_t stream;
    RETURN_IF_HIPSPARSE_ERROR(hipsparseGetStream(handle, &stream));

    // bsrilu0 analysis
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_zbsrilu0_analysis((rocsparse_handle)handle,
                                                          hipDirectionToHCCDirection(dirA),
                                                          mb,
                                                          nnzb,
                                                          (rocsparse_mat_descr)descrA,
                                                          (const rocsparse_double_complex*)bsrValA,
                                                          bsrRowPtrA,
                                                          bsrColIndA,
                                                          blockDim,
                                                          (rocsparse_mat_info)info,
                                                          rocsparse_analysis_policy_force,
                                                          rocsparse_solve_policy_auto,
                                                          pBuffer));

    // Synchronize stream
    RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseSbsrilu02(hipsparseHandle_t         handle,
                                     hipsparseDirection_t      dirA,
                                     int                       mb,
                                     int                       nnzb,
                                     const hipsparseMatDescr_t descrA,
                                     float*                    bsrValA,
                                     const int*                bsrRowPtrA,
                                     const int*                bsrColIndA,
                                     int                       blockDim,
                                     bsrilu02Info_t            info,
                                     hipsparseSolvePolicy_t    policy,
                                     void*                     pBuffer)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_sbsrilu0((rocsparse_handle)handle,
                                                         hipDirectionToHCCDirection(dirA),
                                                         mb,
                                                         nnzb,
                                                         (rocsparse_mat_descr)descrA,
                                                         bsrValA,
                                                         bsrRowPtrA,
                                                         bsrColIndA,
                                                         blockDim,
                                                         (rocsparse_mat_info)info,
                                                         rocsparse_solve_policy_auto,
                                                         pBuffer));
}

hipsparseStatus_t hipsparseDbsrilu02(hipsparseHandle_t         handle,
                                     hipsparseDirection_t      dirA,
                                     int                       mb,
                                     int                       nnzb,
                                     const hipsparseMatDescr_t descrA,
                                     double*                   bsrValA,
                                     const int*                bsrRowPtrA,
                                     const int*                bsrColIndA,
                                     int                       blockDim,
                                     bsrilu02Info_t            info,
                                     hipsparseSolvePolicy_t    policy,
                                     void*                     pBuffer)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_dbsrilu0((rocsparse_handle)handle,
                                                         hipDirectionToHCCDirection(dirA),
                                                         mb,
                                                         nnzb,
                                                         (rocsparse_mat_descr)descrA,
                                                         bsrValA,
                                                         bsrRowPtrA,
                                                         bsrColIndA,
                                                         blockDim,
                                                         (rocsparse_mat_info)info,
                                                         rocsparse_solve_policy_auto,
                                                         pBuffer));
}

hipsparseStatus_t hipsparseCbsrilu02(hipsparseHandle_t         handle,
                                     hipsparseDirection_t      dirA,
                                     int                       mb,
                                     int                       nnzb,
                                     const hipsparseMatDescr_t descrA,
                                     hipComplex*               bsrValA,
                                     const int*                bsrRowPtrA,
                                     const int*                bsrColIndA,
                                     int                       blockDim,
                                     bsrilu02Info_t            info,
                                     hipsparseSolvePolicy_t    policy,
                                     void*                     pBuffer)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_cbsrilu0((rocsparse_handle)handle,
                                                         hipDirectionToHCCDirection(dirA),
                                                         mb,
                                                         nnzb,
                                                         (rocsparse_mat_descr)descrA,
                                                         (rocsparse_float_complex*)bsrValA,
                                                         bsrRowPtrA,
                                                         bsrColIndA,
                                                         blockDim,
                                                         (rocsparse_mat_info)info,
                                                         rocsparse_solve_policy_auto,
                                                         pBuffer));
}

hipsparseStatus_t hipsparseZbsrilu02(hipsparseHandle_t         handle,
                                     hipsparseDirection_t      dirA,
                                     int                       mb,
                                     int                       nnzb,
                                     const hipsparseMatDescr_t descrA,
                                     hipDoubleComplex*         bsrValA,
                                     const int*                bsrRowPtrA,
                                     const int*                bsrColIndA,
                                     int                       blockDim,
                                     bsrilu02Info_t            info,
                                     hipsparseSolvePolicy_t    policy,
                                     void*                     pBuffer)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_zbsrilu0((rocsparse_handle)handle,
                                                         hipDirectionToHCCDirection(dirA),
                                                         mb,
                                                         nnzb,
                                                         (rocsparse_mat_descr)descrA,
                                                         (rocsparse_double_complex*)bsrValA,
                                                         bsrRowPtrA,
                                                         bsrColIndA,
                                                         blockDim,
                                                         (rocsparse_mat_info)info,
                                                         rocsparse_solve_policy_auto,
                                                         pBuffer));
}

hipsparseStatus_t
    hipsparseXcsrilu02_zeroPivot(hipsparseHandle_t handle, csrilu02Info_t info, int* position)
{
    // Obtain stream, to explicitly sync (cusparse csrilu02_zeropivot is blocking)
    hipStream_t stream;
    RETURN_IF_HIPSPARSE_ERROR(hipsparseGetStream(handle, &stream));

    // csrilu0 zero pivot
    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse_csrilu0_zero_pivot((rocsparse_handle)handle, (rocsparse_mat_info)info, position));

    // Synchronize stream
    RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseScsrilu02_numericBoost(
    hipsparseHandle_t handle, csrilu02Info_t info, int enable_boost, double* tol, float* boost_val)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_dscsrilu0_numeric_boost(
        (rocsparse_handle)handle, (rocsparse_mat_info)info, enable_boost, tol, boost_val));
}

hipsparseStatus_t hipsparseDcsrilu02_numericBoost(
    hipsparseHandle_t handle, csrilu02Info_t info, int enable_boost, double* tol, double* boost_val)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_dcsrilu0_numeric_boost(
        (rocsparse_handle)handle, (rocsparse_mat_info)info, enable_boost, tol, boost_val));
}

hipsparseStatus_t hipsparseCcsrilu02_numericBoost(hipsparseHandle_t handle,
                                                  csrilu02Info_t    info,
                                                  int               enable_boost,
                                                  double*           tol,
                                                  hipComplex*       boost_val)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_dccsrilu0_numeric_boost((rocsparse_handle)handle,
                                          (rocsparse_mat_info)info,
                                          enable_boost,
                                          tol,
                                          (rocsparse_float_complex*)boost_val));
}

hipsparseStatus_t hipsparseZcsrilu02_numericBoost(hipsparseHandle_t handle,
                                                  csrilu02Info_t    info,
                                                  int               enable_boost,
                                                  double*           tol,
                                                  hipDoubleComplex* boost_val)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_zcsrilu0_numeric_boost((rocsparse_handle)handle,
                                         (rocsparse_mat_info)info,
                                         enable_boost,
                                         tol,
                                         (rocsparse_double_complex*)boost_val));
}

hipsparseStatus_t hipsparseScsrilu02_bufferSize(hipsparseHandle_t         handle,
                                                int                       m,
                                                int                       nnz,
                                                const hipsparseMatDescr_t descrA,
                                                float*                    csrSortedValA,
                                                const int*                csrSortedRowPtrA,
                                                const int*                csrSortedColIndA,
                                                csrilu02Info_t            info,
                                                int*                      pBufferSizeInBytes)
{
    if(pBufferSizeInBytes == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    size_t           buffer_size;
    rocsparse_status status;

    status = rocsparse_scsrilu0_buffer_size((rocsparse_handle)handle,
                                            m,
                                            nnz,
                                            (rocsparse_mat_descr)descrA,
                                            csrSortedValA,
                                            csrSortedRowPtrA,
                                            csrSortedColIndA,
                                            (rocsparse_mat_info)info,
                                            &buffer_size);

    *pBufferSizeInBytes = (int)buffer_size;

    return rocSPARSEStatusToHIPStatus(status);
}

hipsparseStatus_t hipsparseDcsrilu02_bufferSize(hipsparseHandle_t         handle,
                                                int                       m,
                                                int                       nnz,
                                                const hipsparseMatDescr_t descrA,
                                                double*                   csrSortedValA,
                                                const int*                csrSortedRowPtrA,
                                                const int*                csrSortedColIndA,
                                                csrilu02Info_t            info,
                                                int*                      pBufferSizeInBytes)
{
    if(pBufferSizeInBytes == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    size_t           buffer_size;
    rocsparse_status status;

    status = rocsparse_dcsrilu0_buffer_size((rocsparse_handle)handle,
                                            m,
                                            nnz,
                                            (rocsparse_mat_descr)descrA,
                                            csrSortedValA,
                                            csrSortedRowPtrA,
                                            csrSortedColIndA,
                                            (rocsparse_mat_info)info,
                                            &buffer_size);

    *pBufferSizeInBytes = (int)buffer_size;

    return rocSPARSEStatusToHIPStatus(status);
}

hipsparseStatus_t hipsparseCcsrilu02_bufferSize(hipsparseHandle_t         handle,
                                                int                       m,
                                                int                       nnz,
                                                const hipsparseMatDescr_t descrA,
                                                hipComplex*               csrSortedValA,
                                                const int*                csrSortedRowPtrA,
                                                const int*                csrSortedColIndA,
                                                csrilu02Info_t            info,
                                                int*                      pBufferSizeInBytes)
{
    if(pBufferSizeInBytes == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    size_t           buffer_size;
    rocsparse_status status;

    status = rocsparse_ccsrilu0_buffer_size((rocsparse_handle)handle,
                                            m,
                                            nnz,
                                            (rocsparse_mat_descr)descrA,
                                            (rocsparse_float_complex*)csrSortedValA,
                                            csrSortedRowPtrA,
                                            csrSortedColIndA,
                                            (rocsparse_mat_info)info,
                                            &buffer_size);

    *pBufferSizeInBytes = (int)buffer_size;

    return rocSPARSEStatusToHIPStatus(status);
}

hipsparseStatus_t hipsparseZcsrilu02_bufferSize(hipsparseHandle_t         handle,
                                                int                       m,
                                                int                       nnz,
                                                const hipsparseMatDescr_t descrA,
                                                hipDoubleComplex*         csrSortedValA,
                                                const int*                csrSortedRowPtrA,
                                                const int*                csrSortedColIndA,
                                                csrilu02Info_t            info,
                                                int*                      pBufferSizeInBytes)
{
    if(pBufferSizeInBytes == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    size_t           buffer_size;
    rocsparse_status status;

    status = rocsparse_zcsrilu0_buffer_size((rocsparse_handle)handle,
                                            m,
                                            nnz,
                                            (rocsparse_mat_descr)descrA,
                                            (rocsparse_double_complex*)csrSortedValA,
                                            csrSortedRowPtrA,
                                            csrSortedColIndA,
                                            (rocsparse_mat_info)info,
                                            &buffer_size);

    *pBufferSizeInBytes = (int)buffer_size;

    return rocSPARSEStatusToHIPStatus(status);
}

hipsparseStatus_t hipsparseScsrilu02_bufferSizeExt(hipsparseHandle_t         handle,
                                                   int                       m,
                                                   int                       nnz,
                                                   const hipsparseMatDescr_t descrA,
                                                   float*                    csrSortedValA,
                                                   const int*                csrSortedRowPtrA,
                                                   const int*                csrSortedColIndA,
                                                   csrilu02Info_t            info,
                                                   size_t*                   pBufferSize)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_scsrilu0_buffer_size((rocsparse_handle)handle,
                                                                     m,
                                                                     nnz,
                                                                     (rocsparse_mat_descr)descrA,
                                                                     csrSortedValA,
                                                                     csrSortedRowPtrA,
                                                                     csrSortedColIndA,
                                                                     (rocsparse_mat_info)info,
                                                                     pBufferSize));
}

hipsparseStatus_t hipsparseDcsrilu02_bufferSizeExt(hipsparseHandle_t         handle,
                                                   int                       m,
                                                   int                       nnz,
                                                   const hipsparseMatDescr_t descrA,
                                                   double*                   csrSortedValA,
                                                   const int*                csrSortedRowPtrA,
                                                   const int*                csrSortedColIndA,
                                                   csrilu02Info_t            info,
                                                   size_t*                   pBufferSize)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_dcsrilu0_buffer_size((rocsparse_handle)handle,
                                                                     m,
                                                                     nnz,
                                                                     (rocsparse_mat_descr)descrA,
                                                                     csrSortedValA,
                                                                     csrSortedRowPtrA,
                                                                     csrSortedColIndA,
                                                                     (rocsparse_mat_info)info,
                                                                     pBufferSize));
}

hipsparseStatus_t hipsparseCcsrilu02_bufferSizeExt(hipsparseHandle_t         handle,
                                                   int                       m,
                                                   int                       nnz,
                                                   const hipsparseMatDescr_t descrA,
                                                   hipComplex*               csrSortedValA,
                                                   const int*                csrSortedRowPtrA,
                                                   const int*                csrSortedColIndA,
                                                   csrilu02Info_t            info,
                                                   size_t*                   pBufferSize)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_ccsrilu0_buffer_size((rocsparse_handle)handle,
                                       m,
                                       nnz,
                                       (rocsparse_mat_descr)descrA,
                                       (rocsparse_float_complex*)csrSortedValA,
                                       csrSortedRowPtrA,
                                       csrSortedColIndA,
                                       (rocsparse_mat_info)info,
                                       pBufferSize));
}

hipsparseStatus_t hipsparseZcsrilu02_bufferSizeExt(hipsparseHandle_t         handle,
                                                   int                       m,
                                                   int                       nnz,
                                                   const hipsparseMatDescr_t descrA,
                                                   hipDoubleComplex*         csrSortedValA,
                                                   const int*                csrSortedRowPtrA,
                                                   const int*                csrSortedColIndA,
                                                   csrilu02Info_t            info,
                                                   size_t*                   pBufferSize)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_zcsrilu0_buffer_size((rocsparse_handle)handle,
                                       m,
                                       nnz,
                                       (rocsparse_mat_descr)descrA,
                                       (rocsparse_double_complex*)csrSortedValA,
                                       csrSortedRowPtrA,
                                       csrSortedColIndA,
                                       (rocsparse_mat_info)info,
                                       pBufferSize));
}

hipsparseStatus_t hipsparseScsrilu02_analysis(hipsparseHandle_t         handle,
                                              int                       m,
                                              int                       nnz,
                                              const hipsparseMatDescr_t descrA,
                                              const float*              csrSortedValA,
                                              const int*                csrSortedRowPtrA,
                                              const int*                csrSortedColIndA,
                                              csrilu02Info_t            info,
                                              hipsparseSolvePolicy_t    policy,
                                              void*                     pBuffer)
{
    // Obtain stream, to explicitly sync (cusparse csrilu02_analysis is blocking)
    hipStream_t stream;
    RETURN_IF_HIPSPARSE_ERROR(hipsparseGetStream(handle, &stream));

    // csrilu0 analysis
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_scsrilu0_analysis((rocsparse_handle)handle,
                                                          m,
                                                          nnz,
                                                          (rocsparse_mat_descr)descrA,
                                                          csrSortedValA,
                                                          csrSortedRowPtrA,
                                                          csrSortedColIndA,
                                                          (rocsparse_mat_info)info,
                                                          rocsparse_analysis_policy_force,
                                                          rocsparse_solve_policy_auto,
                                                          pBuffer));

    // Synchronize stream
    RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseDcsrilu02_analysis(hipsparseHandle_t         handle,
                                              int                       m,
                                              int                       nnz,
                                              const hipsparseMatDescr_t descrA,
                                              const double*             csrSortedValA,
                                              const int*                csrSortedRowPtrA,
                                              const int*                csrSortedColIndA,
                                              csrilu02Info_t            info,
                                              hipsparseSolvePolicy_t    policy,
                                              void*                     pBuffer)
{
    // Obtain stream, to explicitly sync (cusparse csrilu02_analysis is blocking)
    hipStream_t stream;
    RETURN_IF_HIPSPARSE_ERROR(hipsparseGetStream(handle, &stream));

    // csrilu0 analysis
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_dcsrilu0_analysis((rocsparse_handle)handle,
                                                          m,
                                                          nnz,
                                                          (rocsparse_mat_descr)descrA,
                                                          csrSortedValA,
                                                          csrSortedRowPtrA,
                                                          csrSortedColIndA,
                                                          (rocsparse_mat_info)info,
                                                          rocsparse_analysis_policy_force,
                                                          rocsparse_solve_policy_auto,
                                                          pBuffer));

    // Synchronize stream
    RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseCcsrilu02_analysis(hipsparseHandle_t         handle,
                                              int                       m,
                                              int                       nnz,
                                              const hipsparseMatDescr_t descrA,
                                              const hipComplex*         csrSortedValA,
                                              const int*                csrSortedRowPtrA,
                                              const int*                csrSortedColIndA,
                                              csrilu02Info_t            info,
                                              hipsparseSolvePolicy_t    policy,
                                              void*                     pBuffer)
{
    // Obtain stream, to explicitly sync (cusparse csrilu02_analysis is blocking)
    hipStream_t stream;
    RETURN_IF_HIPSPARSE_ERROR(hipsparseGetStream(handle, &stream));

    // csrilu0 analysis
    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse_ccsrilu0_analysis((rocsparse_handle)handle,
                                    m,
                                    nnz,
                                    (rocsparse_mat_descr)descrA,
                                    (const rocsparse_float_complex*)csrSortedValA,
                                    csrSortedRowPtrA,
                                    csrSortedColIndA,
                                    (rocsparse_mat_info)info,
                                    rocsparse_analysis_policy_force,
                                    rocsparse_solve_policy_auto,
                                    pBuffer));

    // Synchronize stream
    RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseZcsrilu02_analysis(hipsparseHandle_t         handle,
                                              int                       m,
                                              int                       nnz,
                                              const hipsparseMatDescr_t descrA,
                                              const hipDoubleComplex*   csrSortedValA,
                                              const int*                csrSortedRowPtrA,
                                              const int*                csrSortedColIndA,
                                              csrilu02Info_t            info,
                                              hipsparseSolvePolicy_t    policy,
                                              void*                     pBuffer)
{
    // Obtain stream, to explicitly sync (cusparse csrilu02_analysis is blocking)
    hipStream_t stream;
    RETURN_IF_HIPSPARSE_ERROR(hipsparseGetStream(handle, &stream));

    // csrilu0 analysis
    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse_zcsrilu0_analysis((rocsparse_handle)handle,
                                    m,
                                    nnz,
                                    (rocsparse_mat_descr)descrA,
                                    (const rocsparse_double_complex*)csrSortedValA,
                                    csrSortedRowPtrA,
                                    csrSortedColIndA,
                                    (rocsparse_mat_info)info,
                                    rocsparse_analysis_policy_force,
                                    rocsparse_solve_policy_auto,
                                    pBuffer));

    // Synchronize stream
    RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseScsrilu02(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       nnz,
                                     const hipsparseMatDescr_t descrA,
                                     float*                    csrSortedValA_valM,
                                     /* matrix A values are updated inplace
                                        to be the preconditioner M values */
                                     const int*             csrSortedRowPtrA,
                                     const int*             csrSortedColIndA,
                                     csrilu02Info_t         info,
                                     hipsparseSolvePolicy_t policy,
                                     void*                  pBuffer)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_scsrilu0((rocsparse_handle)handle,
                                                         m,
                                                         nnz,
                                                         (rocsparse_mat_descr)descrA,
                                                         csrSortedValA_valM,
                                                         csrSortedRowPtrA,
                                                         csrSortedColIndA,
                                                         (rocsparse_mat_info)info,
                                                         rocsparse_solve_policy_auto,
                                                         pBuffer));
}

hipsparseStatus_t hipsparseDcsrilu02(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       nnz,
                                     const hipsparseMatDescr_t descrA,
                                     double*                   csrSortedValA_valM,
                                     /* matrix A values are updated inplace
                                        to be the preconditioner M values */
                                     const int*             csrSortedRowPtrA,
                                     const int*             csrSortedColIndA,
                                     csrilu02Info_t         info,
                                     hipsparseSolvePolicy_t policy,
                                     void*                  pBuffer)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_dcsrilu0((rocsparse_handle)handle,
                                                         m,
                                                         nnz,
                                                         (rocsparse_mat_descr)descrA,
                                                         csrSortedValA_valM,
                                                         csrSortedRowPtrA,
                                                         csrSortedColIndA,
                                                         (rocsparse_mat_info)info,
                                                         rocsparse_solve_policy_auto,
                                                         pBuffer));
}

hipsparseStatus_t hipsparseCcsrilu02(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       nnz,
                                     const hipsparseMatDescr_t descrA,
                                     hipComplex*               csrSortedValA_valM,
                                     /* matrix A values are updated inplace
                                        to be the preconditioner M values */
                                     const int*             csrSortedRowPtrA,
                                     const int*             csrSortedColIndA,
                                     csrilu02Info_t         info,
                                     hipsparseSolvePolicy_t policy,
                                     void*                  pBuffer)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_ccsrilu0((rocsparse_handle)handle,
                           m,
                           nnz,
                           (rocsparse_mat_descr)descrA,
                           (rocsparse_float_complex*)csrSortedValA_valM,
                           csrSortedRowPtrA,
                           csrSortedColIndA,
                           (rocsparse_mat_info)info,
                           rocsparse_solve_policy_auto,
                           pBuffer));
}

hipsparseStatus_t hipsparseZcsrilu02(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       nnz,
                                     const hipsparseMatDescr_t descrA,
                                     hipDoubleComplex*         csrSortedValA_valM,
                                     /* matrix A values are updated inplace
                                        to be the preconditioner M values */
                                     const int*             csrSortedRowPtrA,
                                     const int*             csrSortedColIndA,
                                     csrilu02Info_t         info,
                                     hipsparseSolvePolicy_t policy,
                                     void*                  pBuffer)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_zcsrilu0((rocsparse_handle)handle,
                           m,
                           nnz,
                           (rocsparse_mat_descr)descrA,
                           (rocsparse_double_complex*)csrSortedValA_valM,
                           csrSortedRowPtrA,
                           csrSortedColIndA,
                           (rocsparse_mat_info)info,
                           rocsparse_solve_policy_auto,
                           pBuffer));
}

hipsparseStatus_t
    hipsparseXbsric02_zeroPivot(hipsparseHandle_t handle, bsric02Info_t info, int* position)
{
    // Obtain stream, to explicitly sync (cusparse bsric02_zeropivot is blocking)
    hipStream_t stream;
    RETURN_IF_HIPSPARSE_ERROR(hipsparseGetStream(handle, &stream));

    // bsric0 zero pivot
    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse_bsric0_zero_pivot((rocsparse_handle)handle, (rocsparse_mat_info)info, position));

    // Synchronize stream
    RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseSbsric02_bufferSize(hipsparseHandle_t         handle,
                                               hipsparseDirection_t      dirA,
                                               int                       mb,
                                               int                       nnzb,
                                               const hipsparseMatDescr_t descrA,
                                               float*                    bsrValA,
                                               const int*                bsrRowPtrA,
                                               const int*                bsrColIndA,
                                               int                       blockDim,
                                               bsric02Info_t             info,
                                               int*                      pBufferSizeInBytes)
{
    if(pBufferSizeInBytes == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    size_t           buffer_size;
    rocsparse_status status;

    status = rocsparse_sbsric0_buffer_size((rocsparse_handle)handle,
                                           hipDirectionToHCCDirection(dirA),
                                           mb,
                                           nnzb,
                                           (rocsparse_mat_descr)descrA,
                                           bsrValA,
                                           bsrRowPtrA,
                                           bsrColIndA,
                                           blockDim,
                                           (rocsparse_mat_info)info,
                                           &buffer_size);

    *pBufferSizeInBytes = (int)buffer_size;

    return rocSPARSEStatusToHIPStatus(status);
}

hipsparseStatus_t hipsparseDbsric02_bufferSize(hipsparseHandle_t         handle,
                                               hipsparseDirection_t      dirA,
                                               int                       mb,
                                               int                       nnzb,
                                               const hipsparseMatDescr_t descrA,
                                               double*                   bsrValA,
                                               const int*                bsrRowPtrA,
                                               const int*                bsrColIndA,
                                               int                       blockDim,
                                               bsric02Info_t             info,
                                               int*                      pBufferSizeInBytes)
{
    if(pBufferSizeInBytes == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    size_t           buffer_size;
    rocsparse_status status;

    status = rocsparse_dbsric0_buffer_size((rocsparse_handle)handle,
                                           hipDirectionToHCCDirection(dirA),
                                           mb,
                                           nnzb,
                                           (rocsparse_mat_descr)descrA,
                                           bsrValA,
                                           bsrRowPtrA,
                                           bsrColIndA,
                                           blockDim,
                                           (rocsparse_mat_info)info,
                                           &buffer_size);

    *pBufferSizeInBytes = (int)buffer_size;

    return rocSPARSEStatusToHIPStatus(status);
}

hipsparseStatus_t hipsparseCbsric02_bufferSize(hipsparseHandle_t         handle,
                                               hipsparseDirection_t      dirA,
                                               int                       mb,
                                               int                       nnzb,
                                               const hipsparseMatDescr_t descrA,
                                               hipComplex*               bsrValA,
                                               const int*                bsrRowPtrA,
                                               const int*                bsrColIndA,
                                               int                       blockDim,
                                               bsric02Info_t             info,
                                               int*                      pBufferSizeInBytes)
{
    if(pBufferSizeInBytes == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    size_t           buffer_size;
    rocsparse_status status;

    status = rocsparse_cbsric0_buffer_size((rocsparse_handle)handle,
                                           hipDirectionToHCCDirection(dirA),
                                           mb,
                                           nnzb,
                                           (rocsparse_mat_descr)descrA,
                                           (rocsparse_float_complex*)bsrValA,
                                           bsrRowPtrA,
                                           bsrColIndA,
                                           blockDim,
                                           (rocsparse_mat_info)info,
                                           &buffer_size);

    *pBufferSizeInBytes = (int)buffer_size;

    return rocSPARSEStatusToHIPStatus(status);
}

hipsparseStatus_t hipsparseZbsric02_bufferSize(hipsparseHandle_t         handle,
                                               hipsparseDirection_t      dirA,
                                               int                       mb,
                                               int                       nnzb,
                                               const hipsparseMatDescr_t descrA,
                                               hipDoubleComplex*         bsrValA,
                                               const int*                bsrRowPtrA,
                                               const int*                bsrColIndA,
                                               int                       blockDim,
                                               bsric02Info_t             info,
                                               int*                      pBufferSizeInBytes)
{
    if(pBufferSizeInBytes == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    size_t           buffer_size;
    rocsparse_status status;

    status = rocsparse_zbsric0_buffer_size((rocsparse_handle)handle,
                                           hipDirectionToHCCDirection(dirA),
                                           mb,
                                           nnzb,
                                           (rocsparse_mat_descr)descrA,
                                           (rocsparse_double_complex*)bsrValA,
                                           bsrRowPtrA,
                                           bsrColIndA,
                                           blockDim,
                                           (rocsparse_mat_info)info,
                                           &buffer_size);

    *pBufferSizeInBytes = (int)buffer_size;

    return rocSPARSEStatusToHIPStatus(status);
}

hipsparseStatus_t hipsparseSbsric02_analysis(hipsparseHandle_t         handle,
                                             hipsparseDirection_t      dirA,
                                             int                       mb,
                                             int                       nnzb,
                                             const hipsparseMatDescr_t descrA,
                                             const float*              bsrValA,
                                             const int*                bsrRowPtrA,
                                             const int*                bsrColIndA,
                                             int                       blockDim,
                                             bsric02Info_t             info,
                                             hipsparseSolvePolicy_t    policy,
                                             void*                     pBuffer)
{
    // Obtain stream, to explicitly sync (cusparse bsric02_analysis is blocking)
    hipStream_t stream;
    RETURN_IF_HIPSPARSE_ERROR(hipsparseGetStream(handle, &stream));

    // bsric0 analysis
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_sbsric0_analysis((rocsparse_handle)handle,
                                                         hipDirectionToHCCDirection(dirA),
                                                         mb,
                                                         nnzb,
                                                         (rocsparse_mat_descr)descrA,
                                                         bsrValA,
                                                         bsrRowPtrA,
                                                         bsrColIndA,
                                                         blockDim,
                                                         (rocsparse_mat_info)info,
                                                         rocsparse_analysis_policy_force,
                                                         rocsparse_solve_policy_auto,
                                                         pBuffer));

    // Synchronize stream
    RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseDbsric02_analysis(hipsparseHandle_t         handle,
                                             hipsparseDirection_t      dirA,
                                             int                       mb,
                                             int                       nnzb,
                                             const hipsparseMatDescr_t descrA,
                                             const double*             bsrValA,
                                             const int*                bsrRowPtrA,
                                             const int*                bsrColIndA,
                                             int                       blockDim,
                                             bsric02Info_t             info,
                                             hipsparseSolvePolicy_t    policy,
                                             void*                     pBuffer)
{
    // Obtain stream, to explicitly sync (cusparse bsric02_analysis is blocking)
    hipStream_t stream;
    RETURN_IF_HIPSPARSE_ERROR(hipsparseGetStream(handle, &stream));

    // bsric0 analysis
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_dbsric0_analysis((rocsparse_handle)handle,
                                                         hipDirectionToHCCDirection(dirA),
                                                         mb,
                                                         nnzb,
                                                         (rocsparse_mat_descr)descrA,
                                                         bsrValA,
                                                         bsrRowPtrA,
                                                         bsrColIndA,
                                                         blockDim,
                                                         (rocsparse_mat_info)info,
                                                         rocsparse_analysis_policy_force,
                                                         rocsparse_solve_policy_auto,
                                                         pBuffer));

    // Synchronize stream
    RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseCbsric02_analysis(hipsparseHandle_t         handle,
                                             hipsparseDirection_t      dirA,
                                             int                       mb,
                                             int                       nnzb,
                                             const hipsparseMatDescr_t descrA,
                                             const hipComplex*         bsrValA,
                                             const int*                bsrRowPtrA,
                                             const int*                bsrColIndA,
                                             int                       blockDim,
                                             bsric02Info_t             info,
                                             hipsparseSolvePolicy_t    policy,
                                             void*                     pBuffer)
{
    // Obtain stream, to explicitly sync (cusparse bsric02_analysis is blocking)
    hipStream_t stream;
    RETURN_IF_HIPSPARSE_ERROR(hipsparseGetStream(handle, &stream));

    // bsric0 analysis
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_cbsric0_analysis((rocsparse_handle)handle,
                                                         hipDirectionToHCCDirection(dirA),
                                                         mb,
                                                         nnzb,
                                                         (rocsparse_mat_descr)descrA,
                                                         (const rocsparse_float_complex*)bsrValA,
                                                         bsrRowPtrA,
                                                         bsrColIndA,
                                                         blockDim,
                                                         (rocsparse_mat_info)info,
                                                         rocsparse_analysis_policy_force,
                                                         rocsparse_solve_policy_auto,
                                                         pBuffer));

    // Synchronize stream
    RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseZbsric02_analysis(hipsparseHandle_t         handle,
                                             hipsparseDirection_t      dirA,
                                             int                       mb,
                                             int                       nnzb,
                                             const hipsparseMatDescr_t descrA,
                                             const hipDoubleComplex*   bsrValA,
                                             const int*                bsrRowPtrA,
                                             const int*                bsrColIndA,
                                             int                       blockDim,
                                             bsric02Info_t             info,
                                             hipsparseSolvePolicy_t    policy,
                                             void*                     pBuffer)
{
    // Obtain stream, to explicitly sync (cusparse bsric02_analysis is blocking)
    hipStream_t stream;
    RETURN_IF_HIPSPARSE_ERROR(hipsparseGetStream(handle, &stream));

    // bsric0 analysis
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_zbsric0_analysis((rocsparse_handle)handle,
                                                         hipDirectionToHCCDirection(dirA),
                                                         mb,
                                                         nnzb,
                                                         (rocsparse_mat_descr)descrA,
                                                         (const rocsparse_double_complex*)bsrValA,
                                                         bsrRowPtrA,
                                                         bsrColIndA,
                                                         blockDim,
                                                         (rocsparse_mat_info)info,
                                                         rocsparse_analysis_policy_force,
                                                         rocsparse_solve_policy_auto,
                                                         pBuffer));

    // Synchronize stream
    RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseSbsric02(hipsparseHandle_t         handle,
                                    hipsparseDirection_t      dirA,
                                    int                       mb,
                                    int                       nnzb,
                                    const hipsparseMatDescr_t descrA,
                                    float*                    bsrValA,
                                    const int*                bsrRowPtrA,
                                    const int*                bsrColIndA,
                                    int                       blockDim,
                                    bsric02Info_t             info,
                                    hipsparseSolvePolicy_t    policy,
                                    void*                     pBuffer)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_sbsric0((rocsparse_handle)handle,
                                                        hipDirectionToHCCDirection(dirA),
                                                        mb,
                                                        nnzb,
                                                        (rocsparse_mat_descr)descrA,
                                                        bsrValA,
                                                        bsrRowPtrA,
                                                        bsrColIndA,
                                                        blockDim,
                                                        (rocsparse_mat_info)info,
                                                        rocsparse_solve_policy_auto,
                                                        pBuffer));
}

hipsparseStatus_t hipsparseDbsric02(hipsparseHandle_t         handle,
                                    hipsparseDirection_t      dirA,
                                    int                       mb,
                                    int                       nnzb,
                                    const hipsparseMatDescr_t descrA,
                                    double*                   bsrValA,
                                    const int*                bsrRowPtrA,
                                    const int*                bsrColIndA,
                                    int                       blockDim,
                                    bsric02Info_t             info,
                                    hipsparseSolvePolicy_t    policy,
                                    void*                     pBuffer)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_dbsric0((rocsparse_handle)handle,
                                                        hipDirectionToHCCDirection(dirA),
                                                        mb,
                                                        nnzb,
                                                        (rocsparse_mat_descr)descrA,
                                                        bsrValA,
                                                        bsrRowPtrA,
                                                        bsrColIndA,
                                                        blockDim,
                                                        (rocsparse_mat_info)info,
                                                        rocsparse_solve_policy_auto,
                                                        pBuffer));
}

hipsparseStatus_t hipsparseCbsric02(hipsparseHandle_t         handle,
                                    hipsparseDirection_t      dirA,
                                    int                       mb,
                                    int                       nnzb,
                                    const hipsparseMatDescr_t descrA,
                                    hipComplex*               bsrValA,
                                    const int*                bsrRowPtrA,
                                    const int*                bsrColIndA,
                                    int                       blockDim,
                                    bsric02Info_t             info,
                                    hipsparseSolvePolicy_t    policy,
                                    void*                     pBuffer)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_cbsric0((rocsparse_handle)handle,
                                                        hipDirectionToHCCDirection(dirA),
                                                        mb,
                                                        nnzb,
                                                        (rocsparse_mat_descr)descrA,
                                                        (rocsparse_float_complex*)bsrValA,
                                                        bsrRowPtrA,
                                                        bsrColIndA,
                                                        blockDim,
                                                        (rocsparse_mat_info)info,
                                                        rocsparse_solve_policy_auto,
                                                        pBuffer));
}

hipsparseStatus_t hipsparseZbsric02(hipsparseHandle_t         handle,
                                    hipsparseDirection_t      dirA,
                                    int                       mb,
                                    int                       nnzb,
                                    const hipsparseMatDescr_t descrA,
                                    hipDoubleComplex*         bsrValA,
                                    const int*                bsrRowPtrA,
                                    const int*                bsrColIndA,
                                    int                       blockDim,
                                    bsric02Info_t             info,
                                    hipsparseSolvePolicy_t    policy,
                                    void*                     pBuffer)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_zbsric0((rocsparse_handle)handle,
                                                        hipDirectionToHCCDirection(dirA),
                                                        mb,
                                                        nnzb,
                                                        (rocsparse_mat_descr)descrA,
                                                        (rocsparse_double_complex*)bsrValA,
                                                        bsrRowPtrA,
                                                        bsrColIndA,
                                                        blockDim,
                                                        (rocsparse_mat_info)info,
                                                        rocsparse_solve_policy_auto,
                                                        pBuffer));
}

hipsparseStatus_t
    hipsparseXcsric02_zeroPivot(hipsparseHandle_t handle, csric02Info_t info, int* position)
{
    // Obtain stream, to explicitly sync (cusparse csric02_zeropivot is blocking)
    hipStream_t stream;
    RETURN_IF_HIPSPARSE_ERROR(hipsparseGetStream(handle, &stream));

    // csric0 zero pivot
    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse_csric0_zero_pivot((rocsparse_handle)handle, (rocsparse_mat_info)info, position));

    // Synchronize stream
    RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseScsric02_bufferSize(hipsparseHandle_t         handle,
                                               int                       m,
                                               int                       nnz,
                                               const hipsparseMatDescr_t descrA,
                                               float*                    csrSortedValA,
                                               const int*                csrSortedRowPtrA,
                                               const int*                csrSortedColIndA,
                                               csric02Info_t             info,
                                               int*                      pBufferSizeInBytes)
{
    if(pBufferSizeInBytes == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    size_t           buffer_size;
    rocsparse_status status;

    status = rocsparse_scsric0_buffer_size((rocsparse_handle)handle,
                                           m,
                                           nnz,
                                           (rocsparse_mat_descr)descrA,
                                           csrSortedValA,
                                           csrSortedRowPtrA,
                                           csrSortedColIndA,
                                           (rocsparse_mat_info)info,
                                           &buffer_size);

    *pBufferSizeInBytes = (int)buffer_size;

    return rocSPARSEStatusToHIPStatus(status);
}

hipsparseStatus_t hipsparseDcsric02_bufferSize(hipsparseHandle_t         handle,
                                               int                       m,
                                               int                       nnz,
                                               const hipsparseMatDescr_t descrA,
                                               double*                   csrSortedValA,
                                               const int*                csrSortedRowPtrA,
                                               const int*                csrSortedColIndA,
                                               csric02Info_t             info,
                                               int*                      pBufferSizeInBytes)
{
    if(pBufferSizeInBytes == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    size_t           buffer_size;
    rocsparse_status status;

    status = rocsparse_dcsric0_buffer_size((rocsparse_handle)handle,
                                           m,
                                           nnz,
                                           (rocsparse_mat_descr)descrA,
                                           csrSortedValA,
                                           csrSortedRowPtrA,
                                           csrSortedColIndA,
                                           (rocsparse_mat_info)info,
                                           &buffer_size);

    *pBufferSizeInBytes = (int)buffer_size;

    return rocSPARSEStatusToHIPStatus(status);
}

hipsparseStatus_t hipsparseCcsric02_bufferSize(hipsparseHandle_t         handle,
                                               int                       m,
                                               int                       nnz,
                                               const hipsparseMatDescr_t descrA,
                                               hipComplex*               csrSortedValA,
                                               const int*                csrSortedRowPtrA,
                                               const int*                csrSortedColIndA,
                                               csric02Info_t             info,
                                               int*                      pBufferSizeInBytes)
{
    if(pBufferSizeInBytes == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    size_t           buffer_size;
    rocsparse_status status;

    status = rocsparse_ccsric0_buffer_size((rocsparse_handle)handle,
                                           m,
                                           nnz,
                                           (rocsparse_mat_descr)descrA,
                                           (rocsparse_float_complex*)csrSortedValA,
                                           csrSortedRowPtrA,
                                           csrSortedColIndA,
                                           (rocsparse_mat_info)info,
                                           &buffer_size);

    *pBufferSizeInBytes = (int)buffer_size;

    return rocSPARSEStatusToHIPStatus(status);
}

hipsparseStatus_t hipsparseZcsric02_bufferSize(hipsparseHandle_t         handle,
                                               int                       m,
                                               int                       nnz,
                                               const hipsparseMatDescr_t descrA,
                                               hipDoubleComplex*         csrSortedValA,
                                               const int*                csrSortedRowPtrA,
                                               const int*                csrSortedColIndA,
                                               csric02Info_t             info,
                                               int*                      pBufferSizeInBytes)
{
    if(pBufferSizeInBytes == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    size_t           buffer_size;
    rocsparse_status status;

    status = rocsparse_zcsric0_buffer_size((rocsparse_handle)handle,
                                           m,
                                           nnz,
                                           (rocsparse_mat_descr)descrA,
                                           (rocsparse_double_complex*)csrSortedValA,
                                           csrSortedRowPtrA,
                                           csrSortedColIndA,
                                           (rocsparse_mat_info)info,
                                           &buffer_size);

    *pBufferSizeInBytes = (int)buffer_size;

    return rocSPARSEStatusToHIPStatus(status);
}

hipsparseStatus_t hipsparseScsric02_bufferSizeExt(hipsparseHandle_t         handle,
                                                  int                       m,
                                                  int                       nnz,
                                                  const hipsparseMatDescr_t descrA,
                                                  float*                    csrSortedValA,
                                                  const int*                csrSortedRowPtrA,
                                                  const int*                csrSortedColIndA,
                                                  csric02Info_t             info,
                                                  size_t*                   pBufferSize)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_scsric0_buffer_size((rocsparse_handle)handle,
                                                                    m,
                                                                    nnz,
                                                                    (rocsparse_mat_descr)descrA,
                                                                    csrSortedValA,
                                                                    csrSortedRowPtrA,
                                                                    csrSortedColIndA,
                                                                    (rocsparse_mat_info)info,
                                                                    pBufferSize));
}

hipsparseStatus_t hipsparseDcsric02_bufferSizeExt(hipsparseHandle_t         handle,
                                                  int                       m,
                                                  int                       nnz,
                                                  const hipsparseMatDescr_t descrA,
                                                  double*                   csrSortedValA,
                                                  const int*                csrSortedRowPtrA,
                                                  const int*                csrSortedColIndA,
                                                  csric02Info_t             info,
                                                  size_t*                   pBufferSize)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_dcsric0_buffer_size((rocsparse_handle)handle,
                                                                    m,
                                                                    nnz,
                                                                    (rocsparse_mat_descr)descrA,
                                                                    csrSortedValA,
                                                                    csrSortedRowPtrA,
                                                                    csrSortedColIndA,
                                                                    (rocsparse_mat_info)info,
                                                                    pBufferSize));
}

hipsparseStatus_t hipsparseCcsric02_bufferSizeExt(hipsparseHandle_t         handle,
                                                  int                       m,
                                                  int                       nnz,
                                                  const hipsparseMatDescr_t descrA,
                                                  hipComplex*               csrSortedValA,
                                                  const int*                csrSortedRowPtrA,
                                                  const int*                csrSortedColIndA,
                                                  csric02Info_t             info,
                                                  size_t*                   pBufferSize)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_ccsric0_buffer_size((rocsparse_handle)handle,
                                      m,
                                      nnz,
                                      (rocsparse_mat_descr)descrA,
                                      (rocsparse_float_complex*)csrSortedValA,
                                      csrSortedRowPtrA,
                                      csrSortedColIndA,
                                      (rocsparse_mat_info)info,
                                      pBufferSize));
}

hipsparseStatus_t hipsparseZcsric02_bufferSizeExt(hipsparseHandle_t         handle,
                                                  int                       m,
                                                  int                       nnz,
                                                  const hipsparseMatDescr_t descrA,
                                                  hipDoubleComplex*         csrSortedValA,
                                                  const int*                csrSortedRowPtrA,
                                                  const int*                csrSortedColIndA,
                                                  csric02Info_t             info,
                                                  size_t*                   pBufferSize)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_zcsric0_buffer_size((rocsparse_handle)handle,
                                      m,
                                      nnz,
                                      (rocsparse_mat_descr)descrA,
                                      (rocsparse_double_complex*)csrSortedValA,
                                      csrSortedRowPtrA,
                                      csrSortedColIndA,
                                      (rocsparse_mat_info)info,
                                      pBufferSize));
}

hipsparseStatus_t hipsparseScsric02_analysis(hipsparseHandle_t         handle,
                                             int                       m,
                                             int                       nnz,
                                             const hipsparseMatDescr_t descrA,
                                             const float*              csrSortedValA,
                                             const int*                csrSortedRowPtrA,
                                             const int*                csrSortedColIndA,
                                             csric02Info_t             info,
                                             hipsparseSolvePolicy_t    policy,
                                             void*                     pBuffer)
{
    // Obtain stream, to explicitly sync (cusparse csric02_analysis is blocking)
    hipStream_t stream;
    RETURN_IF_HIPSPARSE_ERROR(hipsparseGetStream(handle, &stream));

    // csric0 analysis
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_scsric0_analysis((rocsparse_handle)handle,
                                                         m,
                                                         nnz,
                                                         (rocsparse_mat_descr)descrA,
                                                         csrSortedValA,
                                                         csrSortedRowPtrA,
                                                         csrSortedColIndA,
                                                         (rocsparse_mat_info)info,
                                                         rocsparse_analysis_policy_force,
                                                         rocsparse_solve_policy_auto,
                                                         pBuffer));

    // Synchronize stream
    RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseDcsric02_analysis(hipsparseHandle_t         handle,
                                             int                       m,
                                             int                       nnz,
                                             const hipsparseMatDescr_t descrA,
                                             const double*             csrSortedValA,
                                             const int*                csrSortedRowPtrA,
                                             const int*                csrSortedColIndA,
                                             csric02Info_t             info,
                                             hipsparseSolvePolicy_t    policy,
                                             void*                     pBuffer)
{
    // Obtain stream, to explicitly sync (cusparse csric02_analysis is blocking)
    hipStream_t stream;
    RETURN_IF_HIPSPARSE_ERROR(hipsparseGetStream(handle, &stream));

    // csric0 analysis
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_dcsric0_analysis((rocsparse_handle)handle,
                                                         m,
                                                         nnz,
                                                         (rocsparse_mat_descr)descrA,
                                                         csrSortedValA,
                                                         csrSortedRowPtrA,
                                                         csrSortedColIndA,
                                                         (rocsparse_mat_info)info,
                                                         rocsparse_analysis_policy_force,
                                                         rocsparse_solve_policy_auto,
                                                         pBuffer));

    // Synchronize stream
    RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseCcsric02_analysis(hipsparseHandle_t         handle,
                                             int                       m,
                                             int                       nnz,
                                             const hipsparseMatDescr_t descrA,
                                             const hipComplex*         csrSortedValA,
                                             const int*                csrSortedRowPtrA,
                                             const int*                csrSortedColIndA,
                                             csric02Info_t             info,
                                             hipsparseSolvePolicy_t    policy,
                                             void*                     pBuffer)
{
    // Obtain stream, to explicitly sync (cusparse csric02_analysis is blocking)
    hipStream_t stream;
    RETURN_IF_HIPSPARSE_ERROR(hipsparseGetStream(handle, &stream));

    // csric0 analysis
    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse_ccsric0_analysis((rocsparse_handle)handle,
                                   m,
                                   nnz,
                                   (rocsparse_mat_descr)descrA,
                                   (const rocsparse_float_complex*)csrSortedValA,
                                   csrSortedRowPtrA,
                                   csrSortedColIndA,
                                   (rocsparse_mat_info)info,
                                   rocsparse_analysis_policy_force,
                                   rocsparse_solve_policy_auto,
                                   pBuffer));

    // Synchronize stream
    RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseZcsric02_analysis(hipsparseHandle_t         handle,
                                             int                       m,
                                             int                       nnz,
                                             const hipsparseMatDescr_t descrA,
                                             const hipDoubleComplex*   csrSortedValA,
                                             const int*                csrSortedRowPtrA,
                                             const int*                csrSortedColIndA,
                                             csric02Info_t             info,
                                             hipsparseSolvePolicy_t    policy,
                                             void*                     pBuffer)
{
    // Obtain stream, to explicitly sync (cusparse csric02_analysis is blocking)
    hipStream_t stream;
    RETURN_IF_HIPSPARSE_ERROR(hipsparseGetStream(handle, &stream));

    // csric0 analysis
    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse_zcsric0_analysis((rocsparse_handle)handle,
                                   m,
                                   nnz,
                                   (rocsparse_mat_descr)descrA,
                                   (const rocsparse_double_complex*)csrSortedValA,
                                   csrSortedRowPtrA,
                                   csrSortedColIndA,
                                   (rocsparse_mat_info)info,
                                   rocsparse_analysis_policy_force,
                                   rocsparse_solve_policy_auto,
                                   pBuffer));

    // Synchronize stream
    RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseScsric02(hipsparseHandle_t         handle,
                                    int                       m,
                                    int                       nnz,
                                    const hipsparseMatDescr_t descrA,
                                    float*                    csrSortedValA_valM,
                                    /* matrix A values are updated inplace
                                        to be the preconditioner M values */
                                    const int*             csrSortedRowPtrA,
                                    const int*             csrSortedColIndA,
                                    csric02Info_t          info,
                                    hipsparseSolvePolicy_t policy,
                                    void*                  pBuffer)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_scsric0((rocsparse_handle)handle,
                                                        m,
                                                        nnz,
                                                        (rocsparse_mat_descr)descrA,
                                                        csrSortedValA_valM,
                                                        csrSortedRowPtrA,
                                                        csrSortedColIndA,
                                                        (rocsparse_mat_info)info,
                                                        rocsparse_solve_policy_auto,
                                                        pBuffer));
}

hipsparseStatus_t hipsparseDcsric02(hipsparseHandle_t         handle,
                                    int                       m,
                                    int                       nnz,
                                    const hipsparseMatDescr_t descrA,
                                    double*                   csrSortedValA_valM,
                                    /* matrix A values are updated inplace
                                        to be the preconditioner M values */
                                    const int*             csrSortedRowPtrA,
                                    const int*             csrSortedColIndA,
                                    csric02Info_t          info,
                                    hipsparseSolvePolicy_t policy,
                                    void*                  pBuffer)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_dcsric0((rocsparse_handle)handle,
                                                        m,
                                                        nnz,
                                                        (rocsparse_mat_descr)descrA,
                                                        csrSortedValA_valM,
                                                        csrSortedRowPtrA,
                                                        csrSortedColIndA,
                                                        (rocsparse_mat_info)info,
                                                        rocsparse_solve_policy_auto,
                                                        pBuffer));
}

hipsparseStatus_t hipsparseCcsric02(hipsparseHandle_t         handle,
                                    int                       m,
                                    int                       nnz,
                                    const hipsparseMatDescr_t descrA,
                                    hipComplex*               csrSortedValA_valM,
                                    /* matrix A values are updated inplace
                                        to be the preconditioner M values */
                                    const int*             csrSortedRowPtrA,
                                    const int*             csrSortedColIndA,
                                    csric02Info_t          info,
                                    hipsparseSolvePolicy_t policy,
                                    void*                  pBuffer)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_ccsric0((rocsparse_handle)handle,
                          m,
                          nnz,
                          (rocsparse_mat_descr)descrA,
                          (rocsparse_float_complex*)csrSortedValA_valM,
                          csrSortedRowPtrA,
                          csrSortedColIndA,
                          (rocsparse_mat_info)info,
                          rocsparse_solve_policy_auto,
                          pBuffer));
}

hipsparseStatus_t hipsparseZcsric02(hipsparseHandle_t         handle,
                                    int                       m,
                                    int                       nnz,
                                    const hipsparseMatDescr_t descrA,
                                    hipDoubleComplex*         csrSortedValA_valM,
                                    /* matrix A values are updated inplace
                                        to be the preconditioner M values */
                                    const int*             csrSortedRowPtrA,
                                    const int*             csrSortedColIndA,
                                    csric02Info_t          info,
                                    hipsparseSolvePolicy_t policy,
                                    void*                  pBuffer)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_zcsric0((rocsparse_handle)handle,
                          m,
                          nnz,
                          (rocsparse_mat_descr)descrA,
                          (rocsparse_double_complex*)csrSortedValA_valM,
                          csrSortedRowPtrA,
                          csrSortedColIndA,
                          (rocsparse_mat_info)info,
                          rocsparse_solve_policy_auto,
                          pBuffer));
}

hipsparseStatus_t hipsparseXcsr2coo(hipsparseHandle_t    handle,
                                    const int*           csrRowPtr,
                                    int                  nnz,
                                    int                  m,
                                    int*                 cooRowInd,
                                    hipsparseIndexBase_t idxBase)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_csr2coo(
        (rocsparse_handle)handle, csrRowPtr, nnz, m, cooRowInd, hipBaseToHCCBase(idxBase)));
}

hipsparseStatus_t hipsparseScsr2csc(hipsparseHandle_t    handle,
                                    int                  m,
                                    int                  n,
                                    int                  nnz,
                                    const float*         csrSortedVal,
                                    const int*           csrSortedRowPtr,
                                    const int*           csrSortedColInd,
                                    float*               cscSortedVal,
                                    int*                 cscSortedRowInd,
                                    int*                 cscSortedColPtr,
                                    hipsparseAction_t    copyValues,
                                    hipsparseIndexBase_t idxBase)
{
    // Determine buffer size
    size_t buffer_size = 0;
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_csr2csc_buffer_size((rocsparse_handle)handle,
                                                            m,
                                                            n,
                                                            nnz,
                                                            csrSortedRowPtr,
                                                            csrSortedColInd,
                                                            hipActionToHCCAction(copyValues),
                                                            &buffer_size));

    // Allocate buffer
    void* buffer = nullptr;
    RETURN_IF_HIP_ERROR(hipMalloc(&buffer, buffer_size));

    // Obtain stream, to explicitly sync (cusparse csr2csc is blocking)
    hipStream_t stream;
    RETURN_IF_HIPSPARSE_ERROR(hipsparseGetStream(handle, &stream));

    // Format conversion
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_scsr2csc((rocsparse_handle)handle,
                                                 m,
                                                 n,
                                                 nnz,
                                                 csrSortedVal,
                                                 csrSortedRowPtr,
                                                 csrSortedColInd,
                                                 cscSortedVal,
                                                 cscSortedRowInd,
                                                 cscSortedColPtr,
                                                 hipActionToHCCAction(copyValues),
                                                 hipBaseToHCCBase(idxBase),
                                                 buffer));

    // Free buffer
    RETURN_IF_HIP_ERROR(hipFree(buffer));

    // Synchronize stream
    RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseDcsr2csc(hipsparseHandle_t    handle,
                                    int                  m,
                                    int                  n,
                                    int                  nnz,
                                    const double*        csrSortedVal,
                                    const int*           csrSortedRowPtr,
                                    const int*           csrSortedColInd,
                                    double*              cscSortedVal,
                                    int*                 cscSortedRowInd,
                                    int*                 cscSortedColPtr,
                                    hipsparseAction_t    copyValues,
                                    hipsparseIndexBase_t idxBase)
{
    // Determine buffer size
    size_t buffer_size = 0;
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_csr2csc_buffer_size((rocsparse_handle)handle,
                                                            m,
                                                            n,
                                                            nnz,
                                                            csrSortedRowPtr,
                                                            csrSortedColInd,
                                                            hipActionToHCCAction(copyValues),
                                                            &buffer_size));

    // Allocate buffer
    void* buffer = nullptr;
    RETURN_IF_HIP_ERROR(hipMalloc(&buffer, buffer_size));

    // Obtain stream, to explicitly sync (cusparse csr2csc is blocking)
    hipStream_t stream;
    RETURN_IF_HIPSPARSE_ERROR(hipsparseGetStream(handle, &stream));

    // Format conversion
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_dcsr2csc((rocsparse_handle)handle,
                                                 m,
                                                 n,
                                                 nnz,
                                                 csrSortedVal,
                                                 csrSortedRowPtr,
                                                 csrSortedColInd,
                                                 cscSortedVal,
                                                 cscSortedRowInd,
                                                 cscSortedColPtr,
                                                 hipActionToHCCAction(copyValues),
                                                 hipBaseToHCCBase(idxBase),
                                                 buffer));

    // Free buffer
    RETURN_IF_HIP_ERROR(hipFree(buffer));

    // Synchronize stream
    RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseCcsr2csc(hipsparseHandle_t    handle,
                                    int                  m,
                                    int                  n,
                                    int                  nnz,
                                    const hipComplex*    csrSortedVal,
                                    const int*           csrSortedRowPtr,
                                    const int*           csrSortedColInd,
                                    hipComplex*          cscSortedVal,
                                    int*                 cscSortedRowInd,
                                    int*                 cscSortedColPtr,
                                    hipsparseAction_t    copyValues,
                                    hipsparseIndexBase_t idxBase)
{
    // Determine buffer size
    size_t buffer_size = 0;
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_csr2csc_buffer_size((rocsparse_handle)handle,
                                                            m,
                                                            n,
                                                            nnz,
                                                            csrSortedRowPtr,
                                                            csrSortedColInd,
                                                            hipActionToHCCAction(copyValues),
                                                            &buffer_size));

    // Allocate buffer
    void* buffer = nullptr;
    RETURN_IF_HIP_ERROR(hipMalloc(&buffer, buffer_size));

    // Obtain stream, to explicitly sync (cusparse csr2csc is blocking)
    hipStream_t stream;
    RETURN_IF_HIPSPARSE_ERROR(hipsparseGetStream(handle, &stream));

    // Format conversion
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_ccsr2csc((rocsparse_handle)handle,
                                                 m,
                                                 n,
                                                 nnz,
                                                 (const rocsparse_float_complex*)csrSortedVal,
                                                 csrSortedRowPtr,
                                                 csrSortedColInd,
                                                 (rocsparse_float_complex*)cscSortedVal,
                                                 cscSortedRowInd,
                                                 cscSortedColPtr,
                                                 hipActionToHCCAction(copyValues),
                                                 hipBaseToHCCBase(idxBase),
                                                 buffer));

    // Free buffer
    RETURN_IF_HIP_ERROR(hipFree(buffer));

    // Synchronize stream
    RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseZcsr2csc(hipsparseHandle_t       handle,
                                    int                     m,
                                    int                     n,
                                    int                     nnz,
                                    const hipDoubleComplex* csrSortedVal,
                                    const int*              csrSortedRowPtr,
                                    const int*              csrSortedColInd,
                                    hipDoubleComplex*       cscSortedVal,
                                    int*                    cscSortedRowInd,
                                    int*                    cscSortedColPtr,
                                    hipsparseAction_t       copyValues,
                                    hipsparseIndexBase_t    idxBase)
{
    // Determine buffer size
    size_t buffer_size = 0;
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_csr2csc_buffer_size((rocsparse_handle)handle,
                                                            m,
                                                            n,
                                                            nnz,
                                                            csrSortedRowPtr,
                                                            csrSortedColInd,
                                                            hipActionToHCCAction(copyValues),
                                                            &buffer_size));

    // Allocate buffer
    void* buffer = nullptr;
    RETURN_IF_HIP_ERROR(hipMalloc(&buffer, buffer_size));

    // Obtain stream, to explicitly sync (cusparse csr2csc is blocking)
    hipStream_t stream;
    RETURN_IF_HIPSPARSE_ERROR(hipsparseGetStream(handle, &stream));

    // Format conversion
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_zcsr2csc((rocsparse_handle)handle,
                                                 m,
                                                 n,
                                                 nnz,
                                                 (const rocsparse_double_complex*)csrSortedVal,
                                                 csrSortedRowPtr,
                                                 csrSortedColInd,
                                                 (rocsparse_double_complex*)cscSortedVal,
                                                 cscSortedRowInd,
                                                 cscSortedColPtr,
                                                 hipActionToHCCAction(copyValues),
                                                 hipBaseToHCCBase(idxBase),
                                                 buffer));

    // Free buffer
    RETURN_IF_HIP_ERROR(hipFree(buffer));

    // Synchronize stream
    RETURN_IF_HIP_ERROR(hipStreamSynchronize(stream));

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseScsr2hyb(hipsparseHandle_t         handle,
                                    int                       m,
                                    int                       n,
                                    const hipsparseMatDescr_t descrA,
                                    const float*              csrSortedValA,
                                    const int*                csrSortedRowPtrA,
                                    const int*                csrSortedColIndA,
                                    hipsparseHybMat_t         hybA,
                                    int                       userEllWidth,
                                    hipsparseHybPartition_t   partitionType)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_scsr2hyb((rocsparse_handle)handle,
                                                         m,
                                                         n,
                                                         (rocsparse_mat_descr)descrA,
                                                         csrSortedValA,
                                                         csrSortedRowPtrA,
                                                         csrSortedColIndA,
                                                         (rocsparse_hyb_mat)hybA,
                                                         userEllWidth,
                                                         hipHybPartToHCCHybPart(partitionType)));
}

hipsparseStatus_t hipsparseDcsr2hyb(hipsparseHandle_t         handle,
                                    int                       m,
                                    int                       n,
                                    const hipsparseMatDescr_t descrA,
                                    const double*             csrSortedValA,
                                    const int*                csrSortedRowPtrA,
                                    const int*                csrSortedColIndA,
                                    hipsparseHybMat_t         hybA,
                                    int                       userEllWidth,
                                    hipsparseHybPartition_t   partitionType)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_dcsr2hyb((rocsparse_handle)handle,
                                                         m,
                                                         n,
                                                         (rocsparse_mat_descr)descrA,
                                                         csrSortedValA,
                                                         csrSortedRowPtrA,
                                                         csrSortedColIndA,
                                                         (rocsparse_hyb_mat)hybA,
                                                         userEllWidth,
                                                         hipHybPartToHCCHybPart(partitionType)));
}

hipsparseStatus_t hipsparseCcsr2hyb(hipsparseHandle_t         handle,
                                    int                       m,
                                    int                       n,
                                    const hipsparseMatDescr_t descrA,
                                    const hipComplex*         csrSortedValA,
                                    const int*                csrSortedRowPtrA,
                                    const int*                csrSortedColIndA,
                                    hipsparseHybMat_t         hybA,
                                    int                       userEllWidth,
                                    hipsparseHybPartition_t   partitionType)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_ccsr2hyb((rocsparse_handle)handle,
                           m,
                           n,
                           (rocsparse_mat_descr)descrA,
                           (const rocsparse_float_complex*)csrSortedValA,
                           csrSortedRowPtrA,
                           csrSortedColIndA,
                           (rocsparse_hyb_mat)hybA,
                           userEllWidth,
                           hipHybPartToHCCHybPart(partitionType)));
}

hipsparseStatus_t hipsparseZcsr2hyb(hipsparseHandle_t         handle,
                                    int                       m,
                                    int                       n,
                                    const hipsparseMatDescr_t descrA,
                                    const hipDoubleComplex*   csrSortedValA,
                                    const int*                csrSortedRowPtrA,
                                    const int*                csrSortedColIndA,
                                    hipsparseHybMat_t         hybA,
                                    int                       userEllWidth,
                                    hipsparseHybPartition_t   partitionType)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_zcsr2hyb((rocsparse_handle)handle,
                           m,
                           n,
                           (rocsparse_mat_descr)descrA,
                           (const rocsparse_double_complex*)csrSortedValA,
                           csrSortedRowPtrA,
                           csrSortedColIndA,
                           (rocsparse_hyb_mat)hybA,
                           userEllWidth,
                           hipHybPartToHCCHybPart(partitionType)));
}

hipsparseStatus_t hipsparseSgebsr2gebsc_bufferSize(hipsparseHandle_t handle,
                                                   int               mb,
                                                   int               nb,
                                                   int               nnzb,
                                                   const float*      bsr_val,
                                                   const int*        bsr_row_ptr,
                                                   const int*        bsr_col_ind,
                                                   int               row_block_dim,
                                                   int               col_block_dim,
                                                   size_t*           p_buffer_size)
{
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_sgebsr2gebsc_buffer_size((rocsparse_handle)handle,
                                                                 mb,
                                                                 nb,
                                                                 nnzb,
                                                                 bsr_val,
                                                                 bsr_row_ptr,
                                                                 bsr_col_ind,
                                                                 row_block_dim,
                                                                 col_block_dim,
                                                                 p_buffer_size));
    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseDgebsr2gebsc_bufferSize(hipsparseHandle_t handle,
                                                   int               mb,
                                                   int               nb,
                                                   int               nnzb,
                                                   const double*     bsr_val,
                                                   const int*        bsr_row_ptr,
                                                   const int*        bsr_col_ind,
                                                   int               row_block_dim,
                                                   int               col_block_dim,
                                                   size_t*           p_buffer_size)
{
    //  (const rocsparse_float_complex*)
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_dgebsr2gebsc_buffer_size((rocsparse_handle)handle,
                                                                 mb,
                                                                 nb,
                                                                 nnzb,
                                                                 bsr_val,
                                                                 bsr_row_ptr,
                                                                 bsr_col_ind,
                                                                 row_block_dim,
                                                                 col_block_dim,
                                                                 p_buffer_size));
    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseCgebsr2gebsc_bufferSize(hipsparseHandle_t handle,
                                                   int               mb,
                                                   int               nb,
                                                   int               nnzb,
                                                   const hipComplex* bsr_val,
                                                   const int*        bsr_row_ptr,
                                                   const int*        bsr_col_ind,
                                                   int               row_block_dim,
                                                   int               col_block_dim,
                                                   size_t*           p_buffer_size)
{

    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse_cgebsr2gebsc_buffer_size((rocsparse_handle)handle,
                                           mb,
                                           nb,
                                           nnzb,
                                           (const rocsparse_float_complex*)bsr_val,
                                           bsr_row_ptr,
                                           bsr_col_ind,
                                           row_block_dim,
                                           col_block_dim,
                                           p_buffer_size));
    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseZgebsr2gebsc_bufferSize(hipsparseHandle_t       handle,
                                                   int                     mb,
                                                   int                     nb,
                                                   int                     nnzb,
                                                   const hipDoubleComplex* bsr_val,
                                                   const int*              bsr_row_ptr,
                                                   const int*              bsr_col_ind,
                                                   int                     row_block_dim,
                                                   int                     col_block_dim,
                                                   size_t*                 p_buffer_size)
{

    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse_zgebsr2gebsc_buffer_size((rocsparse_handle)handle,
                                           mb,
                                           nb,
                                           nnzb,
                                           (const rocsparse_double_complex*)bsr_val,
                                           bsr_row_ptr,
                                           bsr_col_ind,
                                           row_block_dim,
                                           col_block_dim,
                                           p_buffer_size));
    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseSgebsr2gebsc(hipsparseHandle_t    handle,
                                        int                  mb,
                                        int                  nb,
                                        int                  nnzb,
                                        const float*         bsr_val,
                                        const int*           bsr_row_ptr,
                                        const int*           bsr_col_ind,
                                        int                  row_block_dim,
                                        int                  col_block_dim,
                                        float*               bsc_val,
                                        int*                 bsc_row_ind,
                                        int*                 bsc_col_ptr,
                                        hipsparseAction_t    copy_values,
                                        hipsparseIndexBase_t idx_base,
                                        void*                temp_buffer)
{
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_sgebsr2gebsc((rocsparse_handle)handle,
                                                     mb,
                                                     nb,
                                                     nnzb,
                                                     bsr_val,
                                                     bsr_row_ptr,
                                                     bsr_col_ind,
                                                     row_block_dim,
                                                     col_block_dim,
                                                     bsc_val,
                                                     bsc_row_ind,
                                                     bsc_col_ptr,
                                                     hipActionToHCCAction(copy_values),
                                                     hipBaseToHCCBase(idx_base),
                                                     temp_buffer));
    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseDgebsr2gebsc(hipsparseHandle_t    handle,
                                        int                  mb,
                                        int                  nb,
                                        int                  nnzb,
                                        const double*        bsr_val,
                                        const int*           bsr_row_ptr,
                                        const int*           bsr_col_ind,
                                        int                  row_block_dim,
                                        int                  col_block_dim,
                                        double*              bsc_val,
                                        int*                 bsc_row_ind,
                                        int*                 bsc_col_ptr,
                                        hipsparseAction_t    copy_values,
                                        hipsparseIndexBase_t idx_base,
                                        void*                temp_buffer)
{

    RETURN_IF_ROCSPARSE_ERROR(rocsparse_dgebsr2gebsc((rocsparse_handle)handle,
                                                     mb,
                                                     nb,
                                                     nnzb,
                                                     bsr_val,
                                                     bsr_row_ptr,
                                                     bsr_col_ind,
                                                     row_block_dim,
                                                     col_block_dim,
                                                     bsc_val,
                                                     bsc_row_ind,
                                                     bsc_col_ptr,
                                                     hipActionToHCCAction(copy_values),
                                                     hipBaseToHCCBase(idx_base),
                                                     temp_buffer));
    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseCgebsr2gebsc(hipsparseHandle_t    handle,
                                        int                  mb,
                                        int                  nb,
                                        int                  nnzb,
                                        const hipComplex*    bsr_val,
                                        const int*           bsr_row_ptr,
                                        const int*           bsr_col_ind,
                                        int                  row_block_dim,
                                        int                  col_block_dim,
                                        hipComplex*          bsc_val,
                                        int*                 bsc_row_ind,
                                        int*                 bsc_col_ptr,
                                        hipsparseAction_t    copy_values,
                                        hipsparseIndexBase_t idx_base,
                                        void*                temp_buffer)
{

    RETURN_IF_ROCSPARSE_ERROR(rocsparse_cgebsr2gebsc((rocsparse_handle)handle,
                                                     mb,
                                                     nb,
                                                     nnzb,
                                                     (const rocsparse_float_complex*)bsr_val,
                                                     bsr_row_ptr,
                                                     bsr_col_ind,
                                                     row_block_dim,
                                                     col_block_dim,
                                                     (rocsparse_float_complex*)bsc_val,
                                                     bsc_row_ind,
                                                     bsc_col_ptr,
                                                     hipActionToHCCAction(copy_values),
                                                     hipBaseToHCCBase(idx_base),
                                                     temp_buffer));
    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseZgebsr2gebsc(hipsparseHandle_t       handle,
                                        int                     mb,
                                        int                     nb,
                                        int                     nnzb,
                                        const hipDoubleComplex* bsr_val,
                                        const int*              bsr_row_ptr,
                                        const int*              bsr_col_ind,
                                        int                     row_block_dim,
                                        int                     col_block_dim,
                                        hipDoubleComplex*       bsc_val,
                                        int*                    bsc_row_ind,
                                        int*                    bsc_col_ptr,
                                        hipsparseAction_t       copy_values,
                                        hipsparseIndexBase_t    idx_base,
                                        void*                   temp_buffer)
{

    RETURN_IF_ROCSPARSE_ERROR(rocsparse_zgebsr2gebsc((rocsparse_handle)handle,
                                                     mb,
                                                     nb,
                                                     nnzb,
                                                     (const rocsparse_double_complex*)bsr_val,
                                                     bsr_row_ptr,
                                                     bsr_col_ind,
                                                     row_block_dim,
                                                     col_block_dim,
                                                     (rocsparse_double_complex*)bsc_val,
                                                     bsc_row_ind,
                                                     bsc_col_ptr,
                                                     hipActionToHCCAction(copy_values),
                                                     hipBaseToHCCBase(idx_base),
                                                     temp_buffer));
    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseScsr2gebsr_bufferSize(hipsparseHandle_t         handle,
                                                 hipsparseDirection_t      dir,
                                                 int                       m,
                                                 int                       n,
                                                 const hipsparseMatDescr_t csr_descr,
                                                 const float*              csr_val,
                                                 const int*                csr_row_ptr,
                                                 const int*                csr_col_ind,
                                                 int                       row_block_dim,
                                                 int                       col_block_dim,
                                                 size_t*                   p_buffer_size)
{
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_scsr2gebsr_buffer_size((rocsparse_handle)handle,
                                                               hipDirectionToHCCDirection(dir),
                                                               m,
                                                               n,
                                                               (const rocsparse_mat_descr)csr_descr,
                                                               csr_val,
                                                               csr_row_ptr,
                                                               csr_col_ind,
                                                               row_block_dim,
                                                               col_block_dim,
                                                               p_buffer_size));
    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseDcsr2gebsr_bufferSize(hipsparseHandle_t         handle,
                                                 hipsparseDirection_t      dir,
                                                 int                       m,
                                                 int                       n,
                                                 const hipsparseMatDescr_t csr_descr,
                                                 const double*             csr_val,
                                                 const int*                csr_row_ptr,
                                                 const int*                csr_col_ind,
                                                 int                       row_block_dim,
                                                 int                       col_block_dim,
                                                 size_t*                   p_buffer_size)
{
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_dcsr2gebsr_buffer_size((rocsparse_handle)handle,
                                                               hipDirectionToHCCDirection(dir),
                                                               m,
                                                               n,
                                                               (const rocsparse_mat_descr)csr_descr,
                                                               csr_val,
                                                               csr_row_ptr,
                                                               csr_col_ind,
                                                               row_block_dim,
                                                               col_block_dim,
                                                               p_buffer_size));
    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseCcsr2gebsr_bufferSize(hipsparseHandle_t         handle,
                                                 hipsparseDirection_t      dir,
                                                 int                       m,
                                                 int                       n,
                                                 const hipsparseMatDescr_t csr_descr,
                                                 const hipComplex*         csr_val,
                                                 const int*                csr_row_ptr,
                                                 const int*                csr_col_ind,
                                                 int                       row_block_dim,
                                                 int                       col_block_dim,
                                                 size_t*                   p_buffer_size)
{
    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse_ccsr2gebsr_buffer_size((rocsparse_handle)handle,
                                         hipDirectionToHCCDirection(dir),
                                         m,
                                         n,
                                         (const rocsparse_mat_descr)csr_descr,
                                         (const rocsparse_float_complex*)csr_val,
                                         csr_row_ptr,
                                         csr_col_ind,
                                         row_block_dim,
                                         col_block_dim,
                                         p_buffer_size));
    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseZcsr2gebsr_bufferSize(hipsparseHandle_t         handle,
                                                 hipsparseDirection_t      dir,
                                                 int                       m,
                                                 int                       n,
                                                 const hipsparseMatDescr_t csr_descr,
                                                 const hipDoubleComplex*   csr_val,
                                                 const int*                csr_row_ptr,
                                                 const int*                csr_col_ind,
                                                 int                       row_block_dim,
                                                 int                       col_block_dim,
                                                 size_t*                   p_buffer_size)
{
    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse_zcsr2gebsr_buffer_size((rocsparse_handle)handle,
                                         hipDirectionToHCCDirection(dir),
                                         m,
                                         n,
                                         (const rocsparse_mat_descr)csr_descr,
                                         (const rocsparse_double_complex*)csr_val,
                                         csr_row_ptr,
                                         csr_col_ind,
                                         row_block_dim,
                                         col_block_dim,
                                         p_buffer_size));
    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseXcsr2gebsrNnz(hipsparseHandle_t         handle,
                                         hipsparseDirection_t      dir,
                                         int                       m,
                                         int                       n,
                                         const hipsparseMatDescr_t csr_descr,
                                         const int*                csr_row_ptr,
                                         const int*                csr_col_ind,
                                         const hipsparseMatDescr_t bsr_descr,
                                         int*                      bsr_row_ptr,
                                         int                       row_block_dim,
                                         int                       col_block_dim,
                                         int*                      bsr_nnz_devhost,
                                         void*                     p_buffer)
{

    RETURN_IF_ROCSPARSE_ERROR(rocsparse_csr2gebsr_nnz((rocsparse_handle)handle,
                                                      hipDirectionToHCCDirection(dir),
                                                      m,
                                                      n,
                                                      (const rocsparse_mat_descr)csr_descr,
                                                      csr_row_ptr,
                                                      csr_col_ind,
                                                      (const rocsparse_mat_descr)bsr_descr,
                                                      bsr_row_ptr,
                                                      row_block_dim,
                                                      col_block_dim,
                                                      bsr_nnz_devhost,
                                                      p_buffer));
    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseScsr2gebsr(hipsparseHandle_t         handle,
                                      hipsparseDirection_t      dir,
                                      int                       m,
                                      int                       n,
                                      const hipsparseMatDescr_t csr_descr,
                                      const float*              csr_val,
                                      const int*                csr_row_ptr,
                                      const int*                csr_col_ind,
                                      const hipsparseMatDescr_t bsr_descr,
                                      float*                    bsr_val,
                                      int*                      bsr_row_ptr,
                                      int*                      bsr_col_ind,
                                      int                       row_block_dim,
                                      int                       col_block_dim,
                                      void*                     p_buffer)
{
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_scsr2gebsr((rocsparse_handle)handle,
                                                   hipDirectionToHCCDirection(dir),
                                                   m,
                                                   n,
                                                   (const rocsparse_mat_descr)csr_descr,
                                                   csr_val,
                                                   csr_row_ptr,
                                                   csr_col_ind,
                                                   (const rocsparse_mat_descr)bsr_descr,
                                                   bsr_val,
                                                   bsr_row_ptr,
                                                   bsr_col_ind,
                                                   row_block_dim,
                                                   col_block_dim,
                                                   p_buffer));
    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseDcsr2gebsr(hipsparseHandle_t         handle,
                                      hipsparseDirection_t      dir,
                                      int                       m,
                                      int                       n,
                                      const hipsparseMatDescr_t csr_descr,
                                      const double*             csr_val,
                                      const int*                csr_row_ptr,
                                      const int*                csr_col_ind,
                                      const hipsparseMatDescr_t bsr_descr,
                                      double*                   bsr_val,
                                      int*                      bsr_row_ptr,
                                      int*                      bsr_col_ind,
                                      int                       row_block_dim,
                                      int                       col_block_dim,
                                      void*                     p_buffer)
{
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_dcsr2gebsr((rocsparse_handle)handle,
                                                   hipDirectionToHCCDirection(dir),
                                                   m,
                                                   n,
                                                   (const rocsparse_mat_descr)csr_descr,
                                                   csr_val,
                                                   csr_row_ptr,
                                                   csr_col_ind,
                                                   (const rocsparse_mat_descr)bsr_descr,
                                                   bsr_val,
                                                   bsr_row_ptr,
                                                   bsr_col_ind,
                                                   row_block_dim,
                                                   col_block_dim,
                                                   p_buffer));
    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseCcsr2gebsr(hipsparseHandle_t         handle,
                                      hipsparseDirection_t      dir,
                                      int                       m,
                                      int                       n,
                                      const hipsparseMatDescr_t csr_descr,
                                      const hipComplex*         csr_val,
                                      const int*                csr_row_ptr,
                                      const int*                csr_col_ind,
                                      const hipsparseMatDescr_t bsr_descr,
                                      hipComplex*               bsr_val,
                                      int*                      bsr_row_ptr,
                                      int*                      bsr_col_ind,
                                      int                       row_block_dim,
                                      int                       col_block_dim,
                                      void*                     p_buffer)
{
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_ccsr2gebsr((rocsparse_handle)handle,
                                                   hipDirectionToHCCDirection(dir),
                                                   m,
                                                   n,
                                                   (const rocsparse_mat_descr)csr_descr,
                                                   (const rocsparse_float_complex*)csr_val,
                                                   csr_row_ptr,
                                                   csr_col_ind,
                                                   (const rocsparse_mat_descr)bsr_descr,
                                                   (rocsparse_float_complex*)bsr_val,
                                                   bsr_row_ptr,
                                                   bsr_col_ind,
                                                   row_block_dim,
                                                   col_block_dim,
                                                   p_buffer));
    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseZcsr2gebsr(hipsparseHandle_t         handle,
                                      hipsparseDirection_t      dir,
                                      int                       m,
                                      int                       n,
                                      const hipsparseMatDescr_t csr_descr,
                                      const hipDoubleComplex*   csr_val,
                                      const int*                csr_row_ptr,
                                      const int*                csr_col_ind,
                                      const hipsparseMatDescr_t bsr_descr,
                                      hipDoubleComplex*         bsr_val,
                                      int*                      bsr_row_ptr,
                                      int*                      bsr_col_ind,
                                      int                       row_block_dim,
                                      int                       col_block_dim,
                                      void*                     p_buffer)
{
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_zcsr2gebsr((rocsparse_handle)handle,
                                                   hipDirectionToHCCDirection(dir),
                                                   m,
                                                   n,
                                                   (const rocsparse_mat_descr)csr_descr,
                                                   (const rocsparse_double_complex*)csr_val,
                                                   csr_row_ptr,
                                                   csr_col_ind,
                                                   (const rocsparse_mat_descr)bsr_descr,
                                                   (rocsparse_double_complex*)bsr_val,
                                                   bsr_row_ptr,
                                                   bsr_col_ind,
                                                   row_block_dim,
                                                   col_block_dim,
                                                   p_buffer));
    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseScsr2bsr(hipsparseHandle_t         handle,
                                    hipsparseDirection_t      dirA,
                                    int                       m,
                                    int                       n,
                                    const hipsparseMatDescr_t descrA,
                                    const float*              csrValA,
                                    const int*                csrRowPtrA,
                                    const int*                csrColIndA,
                                    int                       blockDim,
                                    const hipsparseMatDescr_t descrC,
                                    float*                    bsrValC,
                                    int*                      bsrRowPtrC,
                                    int*                      bsrColIndC)
{
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_scsr2bsr((rocsparse_handle)handle,
                                                 hipDirectionToHCCDirection(dirA),
                                                 m,
                                                 n,
                                                 (const rocsparse_mat_descr)descrA,
                                                 csrValA,
                                                 csrRowPtrA,
                                                 csrColIndA,
                                                 blockDim,
                                                 (const rocsparse_mat_descr)descrC,
                                                 bsrValC,
                                                 bsrRowPtrC,
                                                 bsrColIndC));
    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseDcsr2bsr(hipsparseHandle_t         handle,
                                    hipsparseDirection_t      dirA,
                                    int                       m,
                                    int                       n,
                                    const hipsparseMatDescr_t descrA,
                                    const double*             csrValA,
                                    const int*                csrRowPtrA,
                                    const int*                csrColIndA,
                                    int                       blockDim,
                                    const hipsparseMatDescr_t descrC,
                                    double*                   bsrValC,
                                    int*                      bsrRowPtrC,
                                    int*                      bsrColIndC)
{
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_dcsr2bsr((rocsparse_handle)handle,
                                                 hipDirectionToHCCDirection(dirA),
                                                 m,
                                                 n,
                                                 (const rocsparse_mat_descr)descrA,
                                                 csrValA,
                                                 csrRowPtrA,
                                                 csrColIndA,
                                                 blockDim,
                                                 (const rocsparse_mat_descr)descrC,
                                                 bsrValC,
                                                 bsrRowPtrC,
                                                 bsrColIndC));
    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseCcsr2bsr(hipsparseHandle_t         handle,
                                    hipsparseDirection_t      dirA,
                                    int                       m,
                                    int                       n,
                                    const hipsparseMatDescr_t descrA,
                                    const hipComplex*         csrValA,
                                    const int*                csrRowPtrA,
                                    const int*                csrColIndA,
                                    int                       blockDim,
                                    const hipsparseMatDescr_t descrC,
                                    hipComplex*               bsrValC,
                                    int*                      bsrRowPtrC,
                                    int*                      bsrColIndC)
{
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_ccsr2bsr((rocsparse_handle)handle,
                                                 hipDirectionToHCCDirection(dirA),
                                                 m,
                                                 n,
                                                 (const rocsparse_mat_descr)descrA,
                                                 (rocsparse_float_complex*)csrValA,
                                                 csrRowPtrA,
                                                 csrColIndA,
                                                 blockDim,
                                                 (const rocsparse_mat_descr)descrC,
                                                 (rocsparse_float_complex*)bsrValC,
                                                 bsrRowPtrC,
                                                 bsrColIndC));
    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseZcsr2bsr(hipsparseHandle_t         handle,
                                    hipsparseDirection_t      dirA,
                                    int                       m,
                                    int                       n,
                                    const hipsparseMatDescr_t descrA,
                                    const hipDoubleComplex*   csrValA,
                                    const int*                csrRowPtrA,
                                    const int*                csrColIndA,
                                    int                       blockDim,
                                    const hipsparseMatDescr_t descrC,
                                    hipDoubleComplex*         bsrValC,
                                    int*                      bsrRowPtrC,
                                    int*                      bsrColIndC)
{
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_zcsr2bsr((rocsparse_handle)handle,
                                                 hipDirectionToHCCDirection(dirA),
                                                 m,
                                                 n,
                                                 (const rocsparse_mat_descr)descrA,
                                                 (rocsparse_double_complex*)csrValA,
                                                 csrRowPtrA,
                                                 csrColIndA,
                                                 blockDim,
                                                 (const rocsparse_mat_descr)descrC,
                                                 (rocsparse_double_complex*)bsrValC,
                                                 bsrRowPtrC,
                                                 bsrColIndC));
    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseSbsr2csr(hipsparseHandle_t         handle,
                                    hipsparseDirection_t      dirA,
                                    int                       mb,
                                    int                       nb,
                                    const hipsparseMatDescr_t descrA,
                                    const float*              bsrValA,
                                    const int*                bsrRowPtrA,
                                    const int*                bsrColIndA,
                                    int                       blockDim,
                                    const hipsparseMatDescr_t descrC,
                                    float*                    csrValC,
                                    int*                      csrRowPtrC,
                                    int*                      csrColIndC)
{
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_sbsr2csr((rocsparse_handle)handle,
                                                 hipDirectionToHCCDirection(dirA),
                                                 mb,
                                                 nb,
                                                 (const rocsparse_mat_descr)descrA,
                                                 bsrValA,
                                                 bsrRowPtrA,
                                                 bsrColIndA,
                                                 blockDim,
                                                 (const rocsparse_mat_descr)descrC,
                                                 csrValC,
                                                 csrRowPtrC,
                                                 csrColIndC));
    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseDbsr2csr(hipsparseHandle_t         handle,
                                    hipsparseDirection_t      dirA,
                                    int                       mb,
                                    int                       nb,
                                    const hipsparseMatDescr_t descrA,
                                    const double*             bsrValA,
                                    const int*                bsrRowPtrA,
                                    const int*                bsrColIndA,
                                    int                       blockDim,
                                    const hipsparseMatDescr_t descrC,
                                    double*                   csrValC,
                                    int*                      csrRowPtrC,
                                    int*                      csrColIndC)
{
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_dbsr2csr((rocsparse_handle)handle,
                                                 hipDirectionToHCCDirection(dirA),
                                                 mb,
                                                 nb,
                                                 (const rocsparse_mat_descr)descrA,
                                                 bsrValA,
                                                 bsrRowPtrA,
                                                 bsrColIndA,
                                                 blockDim,
                                                 (const rocsparse_mat_descr)descrC,
                                                 csrValC,
                                                 csrRowPtrC,
                                                 csrColIndC));
    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseCbsr2csr(hipsparseHandle_t         handle,
                                    hipsparseDirection_t      dirA,
                                    int                       mb,
                                    int                       nb,
                                    const hipsparseMatDescr_t descrA,
                                    const hipComplex*         bsrValA,
                                    const int*                bsrRowPtrA,
                                    const int*                bsrColIndA,
                                    int                       blockDim,
                                    const hipsparseMatDescr_t descrC,
                                    hipComplex*               csrValC,
                                    int*                      csrRowPtrC,
                                    int*                      csrColIndC)
{
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_cbsr2csr((rocsparse_handle)handle,
                                                 hipDirectionToHCCDirection(dirA),
                                                 mb,
                                                 nb,
                                                 (const rocsparse_mat_descr)descrA,
                                                 (rocsparse_float_complex*)bsrValA,
                                                 bsrRowPtrA,
                                                 bsrColIndA,
                                                 blockDim,
                                                 (const rocsparse_mat_descr)descrC,
                                                 (rocsparse_float_complex*)csrValC,
                                                 csrRowPtrC,
                                                 csrColIndC));
    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseZbsr2csr(hipsparseHandle_t         handle,
                                    hipsparseDirection_t      dirA,
                                    int                       mb,
                                    int                       nb,
                                    const hipsparseMatDescr_t descrA,
                                    const hipDoubleComplex*   bsrValA,
                                    const int*                bsrRowPtrA,
                                    const int*                bsrColIndA,
                                    int                       blockDim,
                                    const hipsparseMatDescr_t descrC,
                                    hipDoubleComplex*         csrValC,
                                    int*                      csrRowPtrC,
                                    int*                      csrColIndC)
{
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_zbsr2csr((rocsparse_handle)handle,
                                                 hipDirectionToHCCDirection(dirA),
                                                 mb,
                                                 nb,
                                                 (const rocsparse_mat_descr)descrA,
                                                 (rocsparse_double_complex*)bsrValA,
                                                 bsrRowPtrA,
                                                 bsrColIndA,
                                                 blockDim,
                                                 (const rocsparse_mat_descr)descrC,
                                                 (rocsparse_double_complex*)csrValC,
                                                 csrRowPtrC,
                                                 csrColIndC));
    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseSgebsr2csr(hipsparseHandle_t         handle,
                                      hipsparseDirection_t      dirA,
                                      int                       mb,
                                      int                       nb,
                                      const hipsparseMatDescr_t descrA,
                                      const float*              bsrValA,
                                      const int*                bsrRowPtrA,
                                      const int*                bsrColIndA,
                                      int                       rowBlockDim,
                                      int                       colBlockDim,
                                      const hipsparseMatDescr_t descrC,
                                      float*                    csrValC,
                                      int*                      csrRowPtrC,
                                      int*                      csrColIndC)
{
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_sgebsr2csr((rocsparse_handle)handle,
                                                   hipDirectionToHCCDirection(dirA),
                                                   mb,
                                                   nb,
                                                   (const rocsparse_mat_descr)descrA,
                                                   bsrValA,
                                                   bsrRowPtrA,
                                                   bsrColIndA,
                                                   rowBlockDim,
                                                   colBlockDim,
                                                   (const rocsparse_mat_descr)descrC,
                                                   csrValC,
                                                   csrRowPtrC,
                                                   csrColIndC));
    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseDgebsr2csr(hipsparseHandle_t         handle,
                                      hipsparseDirection_t      dirA,
                                      int                       mb,
                                      int                       nb,
                                      const hipsparseMatDescr_t descrA,
                                      const double*             bsrValA,
                                      const int*                bsrRowPtrA,
                                      const int*                bsrColIndA,
                                      int                       rowBlockDim,
                                      int                       colBlockDim,
                                      const hipsparseMatDescr_t descrC,
                                      double*                   csrValC,
                                      int*                      csrRowPtrC,
                                      int*                      csrColIndC)
{
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_dgebsr2csr((rocsparse_handle)handle,
                                                   hipDirectionToHCCDirection(dirA),
                                                   mb,
                                                   nb,
                                                   (const rocsparse_mat_descr)descrA,
                                                   bsrValA,
                                                   bsrRowPtrA,
                                                   bsrColIndA,
                                                   rowBlockDim,
                                                   colBlockDim,
                                                   (const rocsparse_mat_descr)descrC,
                                                   csrValC,
                                                   csrRowPtrC,
                                                   csrColIndC));
    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseCgebsr2csr(hipsparseHandle_t         handle,
                                      hipsparseDirection_t      dirA,
                                      int                       mb,
                                      int                       nb,
                                      const hipsparseMatDescr_t descrA,
                                      const hipComplex*         bsrValA,
                                      const int*                bsrRowPtrA,
                                      const int*                bsrColIndA,
                                      int                       rowBlockDim,
                                      int                       colBlockDim,
                                      const hipsparseMatDescr_t descrC,
                                      hipComplex*               csrValC,
                                      int*                      csrRowPtrC,
                                      int*                      csrColIndC)
{
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_cgebsr2csr((rocsparse_handle)handle,
                                                   hipDirectionToHCCDirection(dirA),
                                                   mb,
                                                   nb,
                                                   (const rocsparse_mat_descr)descrA,
                                                   (rocsparse_float_complex*)bsrValA,
                                                   bsrRowPtrA,
                                                   bsrColIndA,
                                                   rowBlockDim,
                                                   colBlockDim,
                                                   (const rocsparse_mat_descr)descrC,
                                                   (rocsparse_float_complex*)csrValC,
                                                   csrRowPtrC,
                                                   csrColIndC));
    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseZgebsr2csr(hipsparseHandle_t         handle,
                                      hipsparseDirection_t      dirA,
                                      int                       mb,
                                      int                       nb,
                                      const hipsparseMatDescr_t descrA,
                                      const hipDoubleComplex*   bsrValA,
                                      const int*                bsrRowPtrA,
                                      const int*                bsrColIndA,
                                      int                       rowBlockDim,
                                      int                       colBlockDim,
                                      const hipsparseMatDescr_t descrC,
                                      hipDoubleComplex*         csrValC,
                                      int*                      csrRowPtrC,
                                      int*                      csrColIndC)
{
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_zgebsr2csr((rocsparse_handle)handle,
                                                   hipDirectionToHCCDirection(dirA),
                                                   mb,
                                                   nb,
                                                   (const rocsparse_mat_descr)descrA,
                                                   (rocsparse_double_complex*)bsrValA,
                                                   bsrRowPtrA,
                                                   bsrColIndA,
                                                   rowBlockDim,
                                                   colBlockDim,
                                                   (const rocsparse_mat_descr)descrC,
                                                   (rocsparse_double_complex*)csrValC,
                                                   csrRowPtrC,
                                                   csrColIndC));
    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseScsr2csr_compress(hipsparseHandle_t         handle,
                                             int                       m,
                                             int                       n,
                                             const hipsparseMatDescr_t descrA,
                                             const float*              csrValA,
                                             const int*                csrColIndA,
                                             const int*                csrRowPtrA,
                                             int                       nnzA,
                                             const int*                nnzPerRow,
                                             float*                    csrValC,
                                             int*                      csrColIndC,
                                             int*                      csrRowPtrC,
                                             float                     tol)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_scsr2csr_compress((rocsparse_handle)handle,
                                                                  m,
                                                                  n,
                                                                  (const rocsparse_mat_descr)descrA,
                                                                  csrValA,
                                                                  csrRowPtrA,
                                                                  csrColIndA,
                                                                  nnzA,
                                                                  nnzPerRow,
                                                                  csrValC,
                                                                  csrRowPtrC,
                                                                  csrColIndC,
                                                                  tol));
}

hipsparseStatus_t hipsparseDcsr2csr_compress(hipsparseHandle_t         handle,
                                             int                       m,
                                             int                       n,
                                             const hipsparseMatDescr_t descrA,
                                             const double*             csrValA,
                                             const int*                csrColIndA,
                                             const int*                csrRowPtrA,
                                             int                       nnzA,
                                             const int*                nnzPerRow,
                                             double*                   csrValC,
                                             int*                      csrColIndC,
                                             int*                      csrRowPtrC,
                                             double                    tol)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_dcsr2csr_compress((rocsparse_handle)handle,
                                                                  m,
                                                                  n,
                                                                  (const rocsparse_mat_descr)descrA,
                                                                  csrValA,
                                                                  csrRowPtrA,
                                                                  csrColIndA,
                                                                  nnzA,
                                                                  nnzPerRow,
                                                                  csrValC,
                                                                  csrRowPtrC,
                                                                  csrColIndC,
                                                                  tol));
}

hipsparseStatus_t hipsparseCcsr2csr_compress(hipsparseHandle_t         handle,
                                             int                       m,
                                             int                       n,
                                             const hipsparseMatDescr_t descrA,
                                             const hipComplex*         csrValA,
                                             const int*                csrColIndA,
                                             const int*                csrRowPtrA,
                                             int                       nnzA,
                                             const int*                nnzPerRow,
                                             hipComplex*               csrValC,
                                             int*                      csrColIndC,
                                             int*                      csrRowPtrC,
                                             hipComplex                tol)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_ccsr2csr_compress((rocsparse_handle)handle,
                                    m,
                                    n,
                                    (const rocsparse_mat_descr)descrA,
                                    (const rocsparse_float_complex*)csrValA,
                                    csrRowPtrA,
                                    csrColIndA,
                                    nnzA,
                                    nnzPerRow,
                                    (rocsparse_float_complex*)csrValC,
                                    csrRowPtrC,
                                    csrColIndC,
                                    {hipCrealf(tol), hipCimagf(tol)}));
}

hipsparseStatus_t hipsparseZcsr2csr_compress(hipsparseHandle_t         handle,
                                             int                       m,
                                             int                       n,
                                             const hipsparseMatDescr_t descrA,
                                             const hipDoubleComplex*   csrValA,
                                             const int*                csrColIndA,
                                             const int*                csrRowPtrA,
                                             int                       nnzA,
                                             const int*                nnzPerRow,
                                             hipDoubleComplex*         csrValC,
                                             int*                      csrColIndC,
                                             int*                      csrRowPtrC,
                                             hipDoubleComplex          tol)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_zcsr2csr_compress((rocsparse_handle)handle,
                                    m,
                                    n,
                                    (const rocsparse_mat_descr)descrA,
                                    (const rocsparse_double_complex*)csrValA,
                                    csrRowPtrA,
                                    csrColIndA,
                                    nnzA,
                                    nnzPerRow,
                                    (rocsparse_double_complex*)csrValC,
                                    csrRowPtrC,
                                    csrColIndC,
                                    {hipCreal(tol), hipCimag(tol)}));
}

hipsparseStatus_t hipsparseSpruneCsr2csr_bufferSize(hipsparseHandle_t         handle,
                                                    int                       m,
                                                    int                       n,
                                                    int                       nnzA,
                                                    const hipsparseMatDescr_t descrA,
                                                    const float*              csrValA,
                                                    const int*                csrRowPtrA,
                                                    const int*                csrColIndA,
                                                    const float*              threshold,
                                                    const hipsparseMatDescr_t descrC,
                                                    const float*              csrValC,
                                                    const int*                csrRowPtrC,
                                                    const int*                csrColIndC,
                                                    size_t*                   bufferSize)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_sprune_csr2csr_buffer_size((rocsparse_handle)handle,
                                             m,
                                             n,
                                             nnzA,
                                             (const rocsparse_mat_descr)descrA,
                                             csrValA,
                                             csrRowPtrA,
                                             csrColIndA,
                                             threshold,
                                             (const rocsparse_mat_descr)descrC,
                                             csrValC,
                                             csrRowPtrC,
                                             csrColIndC,
                                             bufferSize));
}

hipsparseStatus_t hipsparseDpruneCsr2csr_bufferSize(hipsparseHandle_t         handle,
                                                    int                       m,
                                                    int                       n,
                                                    int                       nnzA,
                                                    const hipsparseMatDescr_t descrA,
                                                    const double*             csrValA,
                                                    const int*                csrRowPtrA,
                                                    const int*                csrColIndA,
                                                    const double*             threshold,
                                                    const hipsparseMatDescr_t descrC,
                                                    const double*             csrValC,
                                                    const int*                csrRowPtrC,
                                                    const int*                csrColIndC,
                                                    size_t*                   bufferSize)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_dprune_csr2csr_buffer_size((rocsparse_handle)handle,
                                             m,
                                             n,
                                             nnzA,
                                             (const rocsparse_mat_descr)descrA,
                                             csrValA,
                                             csrRowPtrA,
                                             csrColIndA,
                                             threshold,
                                             (const rocsparse_mat_descr)descrC,
                                             csrValC,
                                             csrRowPtrC,
                                             csrColIndC,
                                             bufferSize));
}

hipsparseStatus_t hipsparseSpruneCsr2csr_bufferSizeExt(hipsparseHandle_t         handle,
                                                       int                       m,
                                                       int                       n,
                                                       int                       nnzA,
                                                       const hipsparseMatDescr_t descrA,
                                                       const float*              csrValA,
                                                       const int*                csrRowPtrA,
                                                       const int*                csrColIndA,
                                                       const float*              threshold,
                                                       const hipsparseMatDescr_t descrC,
                                                       const float*              csrValC,
                                                       const int*                csrRowPtrC,
                                                       const int*                csrColIndC,
                                                       size_t*                   bufferSize)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_sprune_csr2csr_buffer_size((rocsparse_handle)handle,
                                             m,
                                             n,
                                             nnzA,
                                             (const rocsparse_mat_descr)descrA,
                                             csrValA,
                                             csrRowPtrA,
                                             csrColIndA,
                                             threshold,
                                             (const rocsparse_mat_descr)descrC,
                                             csrValC,
                                             csrRowPtrC,
                                             csrColIndC,
                                             bufferSize));
}

hipsparseStatus_t hipsparseDpruneCsr2csr_bufferSizeExt(hipsparseHandle_t         handle,
                                                       int                       m,
                                                       int                       n,
                                                       int                       nnzA,
                                                       const hipsparseMatDescr_t descrA,
                                                       const double*             csrValA,
                                                       const int*                csrRowPtrA,
                                                       const int*                csrColIndA,
                                                       const double*             threshold,
                                                       const hipsparseMatDescr_t descrC,
                                                       const double*             csrValC,
                                                       const int*                csrRowPtrC,
                                                       const int*                csrColIndC,
                                                       size_t*                   bufferSize)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_dprune_csr2csr_buffer_size((rocsparse_handle)handle,
                                             m,
                                             n,
                                             nnzA,
                                             (const rocsparse_mat_descr)descrA,
                                             csrValA,
                                             csrRowPtrA,
                                             csrColIndA,
                                             threshold,
                                             (const rocsparse_mat_descr)descrC,
                                             csrValC,
                                             csrRowPtrC,
                                             csrColIndC,
                                             bufferSize));
}

hipsparseStatus_t hipsparseSpruneCsr2csrNnz(hipsparseHandle_t         handle,
                                            int                       m,
                                            int                       n,
                                            int                       nnzA,
                                            const hipsparseMatDescr_t descrA,
                                            const float*              csrValA,
                                            const int*                csrRowPtrA,
                                            const int*                csrColIndA,
                                            const float*              threshold,
                                            const hipsparseMatDescr_t descrC,
                                            int*                      csrRowPtrC,
                                            int*                      nnzTotalDevHostPtr,
                                            void*                     buffer)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_sprune_csr2csr_nnz((rocsparse_handle)handle,
                                     m,
                                     n,
                                     nnzA,
                                     (const rocsparse_mat_descr)descrA,
                                     csrValA,
                                     csrRowPtrA,
                                     csrColIndA,
                                     threshold,
                                     (const rocsparse_mat_descr)descrC,
                                     csrRowPtrC,
                                     nnzTotalDevHostPtr,
                                     buffer));
}

hipsparseStatus_t hipsparseDpruneCsr2csrNnz(hipsparseHandle_t         handle,
                                            int                       m,
                                            int                       n,
                                            int                       nnzA,
                                            const hipsparseMatDescr_t descrA,
                                            const double*             csrValA,
                                            const int*                csrRowPtrA,
                                            const int*                csrColIndA,
                                            const double*             threshold,
                                            const hipsparseMatDescr_t descrC,
                                            int*                      csrRowPtrC,
                                            int*                      nnzTotalDevHostPtr,
                                            void*                     buffer)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_dprune_csr2csr_nnz((rocsparse_handle)handle,
                                     m,
                                     n,
                                     nnzA,
                                     (const rocsparse_mat_descr)descrA,
                                     csrValA,
                                     csrRowPtrA,
                                     csrColIndA,
                                     threshold,
                                     (const rocsparse_mat_descr)descrC,
                                     csrRowPtrC,
                                     nnzTotalDevHostPtr,
                                     buffer));
}

hipsparseStatus_t hipsparseSpruneCsr2csr(hipsparseHandle_t         handle,
                                         int                       m,
                                         int                       n,
                                         int                       nnzA,
                                         const hipsparseMatDescr_t descrA,
                                         const float*              csrValA,
                                         const int*                csrRowPtrA,
                                         const int*                csrColIndA,
                                         const float*              threshold,
                                         const hipsparseMatDescr_t descrC,
                                         float*                    csrValC,
                                         const int*                csrRowPtrC,
                                         int*                      csrColIndC,
                                         void*                     buffer)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_sprune_csr2csr((rocsparse_handle)handle,
                                                               m,
                                                               n,
                                                               nnzA,
                                                               (const rocsparse_mat_descr)descrA,
                                                               csrValA,
                                                               csrRowPtrA,
                                                               csrColIndA,
                                                               threshold,
                                                               (const rocsparse_mat_descr)descrC,
                                                               csrValC,
                                                               csrRowPtrC,
                                                               csrColIndC,
                                                               buffer));
}

hipsparseStatus_t hipsparseDpruneCsr2csr(hipsparseHandle_t         handle,
                                         int                       m,
                                         int                       n,
                                         int                       nnzA,
                                         const hipsparseMatDescr_t descrA,
                                         const double*             csrValA,
                                         const int*                csrRowPtrA,
                                         const int*                csrColIndA,
                                         const double*             threshold,
                                         const hipsparseMatDescr_t descrC,
                                         double*                   csrValC,
                                         const int*                csrRowPtrC,
                                         int*                      csrColIndC,
                                         void*                     buffer)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_dprune_csr2csr((rocsparse_handle)handle,
                                                               m,
                                                               n,
                                                               nnzA,
                                                               (const rocsparse_mat_descr)descrA,
                                                               csrValA,
                                                               csrRowPtrA,
                                                               csrColIndA,
                                                               threshold,
                                                               (const rocsparse_mat_descr)descrC,
                                                               csrValC,
                                                               csrRowPtrC,
                                                               csrColIndC,
                                                               buffer));
}

hipsparseStatus_t hipsparseSpruneCsr2csrByPercentage_bufferSize(hipsparseHandle_t         handle,
                                                                int                       m,
                                                                int                       n,
                                                                int                       nnzA,
                                                                const hipsparseMatDescr_t descrA,
                                                                const float*              csrValA,
                                                                const int* csrRowPtrA,
                                                                const int* csrColIndA,
                                                                float      percentage,
                                                                const hipsparseMatDescr_t descrC,
                                                                const float*              csrValC,
                                                                const int*  csrRowPtrC,
                                                                const int*  csrColIndC,
                                                                pruneInfo_t info,
                                                                size_t*     bufferSize)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_sprune_csr2csr_by_percentage_buffer_size((rocsparse_handle)handle,
                                                           m,
                                                           n,
                                                           nnzA,
                                                           (const rocsparse_mat_descr)descrA,
                                                           csrValA,
                                                           csrRowPtrA,
                                                           csrColIndA,
                                                           percentage,
                                                           (const rocsparse_mat_descr)descrC,
                                                           csrValC,
                                                           csrRowPtrC,
                                                           csrColIndC,
                                                           (rocsparse_mat_info)info,
                                                           bufferSize));
}

hipsparseStatus_t hipsparseDpruneCsr2csrByPercentage_bufferSize(hipsparseHandle_t         handle,
                                                                int                       m,
                                                                int                       n,
                                                                int                       nnzA,
                                                                const hipsparseMatDescr_t descrA,
                                                                const double*             csrValA,
                                                                const int* csrRowPtrA,
                                                                const int* csrColIndA,
                                                                double     percentage,
                                                                const hipsparseMatDescr_t descrC,
                                                                const double*             csrValC,
                                                                const int*  csrRowPtrC,
                                                                const int*  csrColIndC,
                                                                pruneInfo_t info,
                                                                size_t*     bufferSize)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_dprune_csr2csr_by_percentage_buffer_size((rocsparse_handle)handle,
                                                           m,
                                                           n,
                                                           nnzA,
                                                           (const rocsparse_mat_descr)descrA,
                                                           csrValA,
                                                           csrRowPtrA,
                                                           csrColIndA,
                                                           percentage,
                                                           (const rocsparse_mat_descr)descrC,
                                                           csrValC,
                                                           csrRowPtrC,
                                                           csrColIndC,
                                                           (rocsparse_mat_info)info,
                                                           bufferSize));
}

hipsparseStatus_t hipsparseSpruneCsr2csrByPercentage_bufferSizeExt(hipsparseHandle_t         handle,
                                                                   int                       m,
                                                                   int                       n,
                                                                   int                       nnzA,
                                                                   const hipsparseMatDescr_t descrA,
                                                                   const float* csrValA,
                                                                   const int*   csrRowPtrA,
                                                                   const int*   csrColIndA,
                                                                   float        percentage,
                                                                   const hipsparseMatDescr_t descrC,
                                                                   const float* csrValC,
                                                                   const int*   csrRowPtrC,
                                                                   const int*   csrColIndC,
                                                                   pruneInfo_t  info,
                                                                   size_t*      bufferSize)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_sprune_csr2csr_by_percentage_buffer_size((rocsparse_handle)handle,
                                                           m,
                                                           n,
                                                           nnzA,
                                                           (const rocsparse_mat_descr)descrA,
                                                           csrValA,
                                                           csrRowPtrA,
                                                           csrColIndA,
                                                           percentage,
                                                           (const rocsparse_mat_descr)descrC,
                                                           csrValC,
                                                           csrRowPtrC,
                                                           csrColIndC,
                                                           (rocsparse_mat_info)info,
                                                           bufferSize));
}

hipsparseStatus_t hipsparseDpruneCsr2csrByPercentage_bufferSizeExt(hipsparseHandle_t         handle,
                                                                   int                       m,
                                                                   int                       n,
                                                                   int                       nnzA,
                                                                   const hipsparseMatDescr_t descrA,
                                                                   const double* csrValA,
                                                                   const int*    csrRowPtrA,
                                                                   const int*    csrColIndA,
                                                                   double        percentage,
                                                                   const hipsparseMatDescr_t descrC,
                                                                   const double* csrValC,
                                                                   const int*    csrRowPtrC,
                                                                   const int*    csrColIndC,
                                                                   pruneInfo_t   info,
                                                                   size_t*       bufferSize)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_dprune_csr2csr_by_percentage_buffer_size((rocsparse_handle)handle,
                                                           m,
                                                           n,
                                                           nnzA,
                                                           (const rocsparse_mat_descr)descrA,
                                                           csrValA,
                                                           csrRowPtrA,
                                                           csrColIndA,
                                                           percentage,
                                                           (const rocsparse_mat_descr)descrC,
                                                           csrValC,
                                                           csrRowPtrC,
                                                           csrColIndC,
                                                           (rocsparse_mat_info)info,
                                                           bufferSize));
}

hipsparseStatus_t hipsparseSpruneCsr2csrNnzByPercentage(hipsparseHandle_t         handle,
                                                        int                       m,
                                                        int                       n,
                                                        int                       nnzA,
                                                        const hipsparseMatDescr_t descrA,
                                                        const float*              csrValA,
                                                        const int*                csrRowPtrA,
                                                        const int*                csrColIndA,
                                                        float                     percentage,
                                                        const hipsparseMatDescr_t descrC,
                                                        int*                      csrRowPtrC,
                                                        int*        nnzTotalDevHostPtr,
                                                        pruneInfo_t info,
                                                        void*       buffer)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_sprune_csr2csr_nnz_by_percentage((rocsparse_handle)handle,
                                                   m,
                                                   n,
                                                   nnzA,
                                                   (const rocsparse_mat_descr)descrA,
                                                   csrValA,
                                                   csrRowPtrA,
                                                   csrColIndA,
                                                   percentage,
                                                   (const rocsparse_mat_descr)descrC,
                                                   csrRowPtrC,
                                                   nnzTotalDevHostPtr,
                                                   (rocsparse_mat_info)info,
                                                   buffer));
}

hipsparseStatus_t hipsparseDpruneCsr2csrNnzByPercentage(hipsparseHandle_t         handle,
                                                        int                       m,
                                                        int                       n,
                                                        int                       nnzA,
                                                        const hipsparseMatDescr_t descrA,
                                                        const double*             csrValA,
                                                        const int*                csrRowPtrA,
                                                        const int*                csrColIndA,
                                                        double                    percentage,
                                                        const hipsparseMatDescr_t descrC,
                                                        int*                      csrRowPtrC,
                                                        int*        nnzTotalDevHostPtr,
                                                        pruneInfo_t info,
                                                        void*       buffer)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_dprune_csr2csr_nnz_by_percentage((rocsparse_handle)handle,
                                                   m,
                                                   n,
                                                   nnzA,
                                                   (const rocsparse_mat_descr)descrA,
                                                   csrValA,
                                                   csrRowPtrA,
                                                   csrColIndA,
                                                   percentage,
                                                   (const rocsparse_mat_descr)descrC,
                                                   csrRowPtrC,
                                                   nnzTotalDevHostPtr,
                                                   (rocsparse_mat_info)info,
                                                   buffer));
}

hipsparseStatus_t hipsparseSpruneCsr2csrByPercentage(hipsparseHandle_t         handle,
                                                     int                       m,
                                                     int                       n,
                                                     int                       nnzA,
                                                     const hipsparseMatDescr_t descrA,
                                                     const float*              csrValA,
                                                     const int*                csrRowPtrA,
                                                     const int*                csrColIndA,
                                                     float                     percentage,
                                                     const hipsparseMatDescr_t descrC,
                                                     float*                    csrValC,
                                                     const int*                csrRowPtrC,
                                                     int*                      csrColIndC,
                                                     pruneInfo_t               info,
                                                     void*                     buffer)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_sprune_csr2csr_by_percentage((rocsparse_handle)handle,
                                               m,
                                               n,
                                               nnzA,
                                               (const rocsparse_mat_descr)descrA,
                                               csrValA,
                                               csrRowPtrA,
                                               csrColIndA,
                                               percentage,
                                               (const rocsparse_mat_descr)descrC,
                                               csrValC,
                                               csrRowPtrC,
                                               csrColIndC,
                                               (rocsparse_mat_info)info,
                                               buffer));
}

hipsparseStatus_t hipsparseDpruneCsr2csrByPercentage(hipsparseHandle_t         handle,
                                                     int                       m,
                                                     int                       n,
                                                     int                       nnzA,
                                                     const hipsparseMatDescr_t descrA,
                                                     const double*             csrValA,
                                                     const int*                csrRowPtrA,
                                                     const int*                csrColIndA,
                                                     double                    percentage,
                                                     const hipsparseMatDescr_t descrC,
                                                     double*                   csrValC,
                                                     const int*                csrRowPtrC,
                                                     int*                      csrColIndC,
                                                     pruneInfo_t               info,
                                                     void*                     buffer)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_dprune_csr2csr_by_percentage((rocsparse_handle)handle,
                                               m,
                                               n,
                                               nnzA,
                                               (const rocsparse_mat_descr)descrA,
                                               csrValA,
                                               csrRowPtrA,
                                               csrColIndA,
                                               percentage,
                                               (const rocsparse_mat_descr)descrC,
                                               csrValC,
                                               csrRowPtrC,
                                               csrColIndC,
                                               (rocsparse_mat_info)info,
                                               buffer));
}

hipsparseStatus_t hipsparseShyb2csr(hipsparseHandle_t         handle,
                                    const hipsparseMatDescr_t descrA,
                                    const hipsparseHybMat_t   hybA,
                                    float*                    csrSortedValA,
                                    int*                      csrSortedRowPtrA,
                                    int*                      csrSortedColIndA)
{
    // Determine buffer size
    size_t buffer_size = 0;
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_hyb2csr_buffer_size((rocsparse_handle)handle,
                                                            (const rocsparse_mat_descr)descrA,
                                                            (const rocsparse_hyb_mat)hybA,
                                                            csrSortedRowPtrA,
                                                            &buffer_size));

    // Allocate buffer
    void* buffer = nullptr;
    RETURN_IF_HIP_ERROR(hipMalloc(&buffer, buffer_size));

    // Format conversion
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_shyb2csr((rocsparse_handle)handle,
                                                 (const rocsparse_mat_descr)descrA,
                                                 (const rocsparse_hyb_mat)hybA,
                                                 csrSortedValA,
                                                 csrSortedRowPtrA,
                                                 csrSortedColIndA,
                                                 buffer));

    // Free buffer
    RETURN_IF_HIP_ERROR(hipFree(buffer));

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseDhyb2csr(hipsparseHandle_t         handle,
                                    const hipsparseMatDescr_t descrA,
                                    const hipsparseHybMat_t   hybA,
                                    double*                   csrSortedValA,
                                    int*                      csrSortedRowPtrA,
                                    int*                      csrSortedColIndA)
{
    // Determine buffer size
    size_t buffer_size = 0;
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_hyb2csr_buffer_size((rocsparse_handle)handle,
                                                            (const rocsparse_mat_descr)descrA,
                                                            (const rocsparse_hyb_mat)hybA,
                                                            csrSortedRowPtrA,
                                                            &buffer_size));

    // Allocate buffer
    void* buffer = nullptr;
    RETURN_IF_HIP_ERROR(hipMalloc(&buffer, buffer_size));

    // Format conversion
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_dhyb2csr((rocsparse_handle)handle,
                                                 (const rocsparse_mat_descr)descrA,
                                                 (const rocsparse_hyb_mat)hybA,
                                                 csrSortedValA,
                                                 csrSortedRowPtrA,
                                                 csrSortedColIndA,
                                                 buffer));

    // Free buffer
    RETURN_IF_HIP_ERROR(hipFree(buffer));

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseChyb2csr(hipsparseHandle_t         handle,
                                    const hipsparseMatDescr_t descrA,
                                    const hipsparseHybMat_t   hybA,
                                    hipComplex*               csrSortedValA,
                                    int*                      csrSortedRowPtrA,
                                    int*                      csrSortedColIndA)
{
    // Determine buffer size
    size_t buffer_size = 0;
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_hyb2csr_buffer_size((rocsparse_handle)handle,
                                                            (const rocsparse_mat_descr)descrA,
                                                            (const rocsparse_hyb_mat)hybA,
                                                            csrSortedRowPtrA,
                                                            &buffer_size));

    // Allocate buffer
    void* buffer = nullptr;
    RETURN_IF_HIP_ERROR(hipMalloc(&buffer, buffer_size));

    // Format conversion
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_chyb2csr((rocsparse_handle)handle,
                                                 (const rocsparse_mat_descr)descrA,
                                                 (const rocsparse_hyb_mat)hybA,
                                                 (rocsparse_float_complex*)csrSortedValA,
                                                 csrSortedRowPtrA,
                                                 csrSortedColIndA,
                                                 buffer));

    // Free buffer
    RETURN_IF_HIP_ERROR(hipFree(buffer));

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseZhyb2csr(hipsparseHandle_t         handle,
                                    const hipsparseMatDescr_t descrA,
                                    const hipsparseHybMat_t   hybA,
                                    hipDoubleComplex*         csrSortedValA,
                                    int*                      csrSortedRowPtrA,
                                    int*                      csrSortedColIndA)
{
    // Determine buffer size
    size_t buffer_size = 0;
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_hyb2csr_buffer_size((rocsparse_handle)handle,
                                                            (const rocsparse_mat_descr)descrA,
                                                            (const rocsparse_hyb_mat)hybA,
                                                            csrSortedRowPtrA,
                                                            &buffer_size));

    // Allocate buffer
    void* buffer = nullptr;
    RETURN_IF_HIP_ERROR(hipMalloc(&buffer, buffer_size));

    // Format conversion
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_zhyb2csr((rocsparse_handle)handle,
                                                 (const rocsparse_mat_descr)descrA,
                                                 (const rocsparse_hyb_mat)hybA,
                                                 (rocsparse_double_complex*)csrSortedValA,
                                                 csrSortedRowPtrA,
                                                 csrSortedColIndA,
                                                 buffer));

    // Free buffer
    RETURN_IF_HIP_ERROR(hipFree(buffer));

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseSnnz(hipsparseHandle_t         handle,
                                hipsparseDirection_t      dirA,
                                int                       m,
                                int                       n,
                                const hipsparseMatDescr_t descrA,
                                const float*              A,
                                int                       lda,
                                int*                      nnzPerRowColumn,
                                int*                      nnzTotalDevHostPtr)
{
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_snnz((rocsparse_handle)handle,
                                             hipDirectionToHCCDirection(dirA),
                                             m,
                                             n,
                                             (const rocsparse_mat_descr)descrA,
                                             A,
                                             lda,
                                             nnzPerRowColumn,
                                             nnzTotalDevHostPtr));
    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseDnnz(hipsparseHandle_t         handle,
                                hipsparseDirection_t      dirA,
                                int                       m,
                                int                       n,
                                const hipsparseMatDescr_t descrA,
                                const double*             A,
                                int                       lda,
                                int*                      nnzPerRowColumn,
                                int*                      nnzTotalDevHostPtr)
{
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_dnnz((rocsparse_handle)handle,
                                             hipDirectionToHCCDirection(dirA),
                                             m,
                                             n,
                                             (const rocsparse_mat_descr)descrA,
                                             A,
                                             lda,
                                             nnzPerRowColumn,
                                             nnzTotalDevHostPtr));
    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseCnnz(hipsparseHandle_t         handle,
                                hipsparseDirection_t      dirA,
                                int                       m,
                                int                       n,
                                const hipsparseMatDescr_t descrA,
                                const hipComplex*         A,
                                int                       lda,
                                int*                      nnzPerRowColumn,
                                int*                      nnzTotalDevHostPtr)
{
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_cnnz((rocsparse_handle)handle,
                                             hipDirectionToHCCDirection(dirA),
                                             m,
                                             n,
                                             (const rocsparse_mat_descr)descrA,
                                             (const rocsparse_float_complex*)A,
                                             lda,
                                             nnzPerRowColumn,
                                             nnzTotalDevHostPtr));
    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseZnnz(hipsparseHandle_t         handle,
                                hipsparseDirection_t      dirA,
                                int                       m,
                                int                       n,
                                const hipsparseMatDescr_t descrA,
                                const hipDoubleComplex*   A,
                                int                       lda,
                                int*                      nnzPerRowColumn,
                                int*                      nnzTotalDevHostPtr)
{
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_znnz((rocsparse_handle)handle,
                                             hipDirectionToHCCDirection(dirA),
                                             m,
                                             n,
                                             (const rocsparse_mat_descr)descrA,
                                             (const rocsparse_double_complex*)A,
                                             lda,
                                             nnzPerRowColumn,
                                             nnzTotalDevHostPtr));
    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseSdense2csr(hipsparseHandle_t         handle,
                                      int                       m,
                                      int                       n,
                                      const hipsparseMatDescr_t descr,
                                      const float*              A,
                                      int                       ld,
                                      const int*                nnzPerRow,
                                      float*                    csrVal,
                                      int*                      csrRowPtr,
                                      int*                      csrColInd)
{
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_sdense2csr((rocsparse_handle)handle,
                                                   m,
                                                   n,
                                                   (const rocsparse_mat_descr)descr,
                                                   A,
                                                   ld,
                                                   nnzPerRow,
                                                   csrVal,
                                                   csrRowPtr,
                                                   csrColInd));
    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseDdense2csr(hipsparseHandle_t         handle,
                                      int                       m,
                                      int                       n,
                                      const hipsparseMatDescr_t descr,
                                      const double*             A,
                                      int                       ld,
                                      const int*                nnzPerRow,
                                      double*                   csrVal,
                                      int*                      csrRowPtr,
                                      int*                      csrColInd)
{
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_ddense2csr((rocsparse_handle)handle,
                                                   m,
                                                   n,
                                                   (const rocsparse_mat_descr)descr,
                                                   A,
                                                   ld,
                                                   nnzPerRow,
                                                   csrVal,
                                                   csrRowPtr,
                                                   csrColInd));
    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseCdense2csr(hipsparseHandle_t         handle,
                                      int                       m,
                                      int                       n,
                                      const hipsparseMatDescr_t descr,
                                      const hipComplex*         A,
                                      int                       ld,
                                      const int*                nnzPerRow,
                                      hipComplex*               csrVal,
                                      int*                      csrRowPtr,
                                      int*                      csrColInd)
{
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_cdense2csr((rocsparse_handle)handle,
                                                   m,
                                                   n,
                                                   (const rocsparse_mat_descr)descr,
                                                   (const rocsparse_float_complex*)A,
                                                   ld,
                                                   nnzPerRow,
                                                   (rocsparse_float_complex*)csrVal,
                                                   csrRowPtr,
                                                   csrColInd));
    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseZdense2csr(hipsparseHandle_t         handle,
                                      int                       m,
                                      int                       n,
                                      const hipsparseMatDescr_t descr,
                                      const hipDoubleComplex*   A,
                                      int                       ld,
                                      const int*                nnzPerRow,
                                      hipDoubleComplex*         csrVal,
                                      int*                      csrRowPtr,
                                      int*                      csrColInd)
{
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_zdense2csr((rocsparse_handle)handle,
                                                   m,
                                                   n,
                                                   (const rocsparse_mat_descr)descr,
                                                   (const rocsparse_double_complex*)A,
                                                   ld,
                                                   nnzPerRow,
                                                   (rocsparse_double_complex*)csrVal,
                                                   csrRowPtr,
                                                   csrColInd));
    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseSpruneDense2csr_bufferSize(hipsparseHandle_t         handle,
                                                      int                       m,
                                                      int                       n,
                                                      const float*              A,
                                                      int                       lda,
                                                      const float*              threshold,
                                                      const hipsparseMatDescr_t descr,
                                                      const float*              csrVal,
                                                      const int*                csrRowPtr,
                                                      const int*                csrColInd,
                                                      size_t*                   bufferSize)
{
    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse_sprune_dense2csr_buffer_size((rocsparse_handle)handle,
                                               m,
                                               n,
                                               A,
                                               lda,
                                               threshold,
                                               (const rocsparse_mat_descr)descr,
                                               csrVal,
                                               csrRowPtr,
                                               csrColInd,
                                               bufferSize));

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseDpruneDense2csr_bufferSize(hipsparseHandle_t         handle,
                                                      int                       m,
                                                      int                       n,
                                                      const double*             A,
                                                      int                       lda,
                                                      const double*             threshold,
                                                      const hipsparseMatDescr_t descr,
                                                      const double*             csrVal,
                                                      const int*                csrRowPtr,
                                                      const int*                csrColInd,
                                                      size_t*                   bufferSize)
{
    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse_dprune_dense2csr_buffer_size((rocsparse_handle)handle,
                                               m,
                                               n,
                                               A,
                                               lda,
                                               threshold,
                                               (const rocsparse_mat_descr)descr,
                                               csrVal,
                                               csrRowPtr,
                                               csrColInd,
                                               bufferSize));

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseSpruneDense2csr_bufferSizeExt(hipsparseHandle_t         handle,
                                                         int                       m,
                                                         int                       n,
                                                         const float*              A,
                                                         int                       lda,
                                                         const float*              threshold,
                                                         const hipsparseMatDescr_t descr,
                                                         const float*              csrVal,
                                                         const int*                csrRowPtr,
                                                         const int*                csrColInd,
                                                         size_t*                   bufferSize)
{
    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse_sprune_dense2csr_buffer_size((rocsparse_handle)handle,
                                               m,
                                               n,
                                               A,
                                               lda,
                                               threshold,
                                               (const rocsparse_mat_descr)descr,
                                               csrVal,
                                               csrRowPtr,
                                               csrColInd,
                                               bufferSize));

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseDpruneDense2csr_bufferSizeExt(hipsparseHandle_t         handle,
                                                         int                       m,
                                                         int                       n,
                                                         const double*             A,
                                                         int                       lda,
                                                         const double*             threshold,
                                                         const hipsparseMatDescr_t descr,
                                                         const double*             csrVal,
                                                         const int*                csrRowPtr,
                                                         const int*                csrColInd,
                                                         size_t*                   bufferSize)
{
    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse_dprune_dense2csr_buffer_size((rocsparse_handle)handle,
                                               m,
                                               n,
                                               A,
                                               lda,
                                               threshold,
                                               (const rocsparse_mat_descr)descr,
                                               csrVal,
                                               csrRowPtr,
                                               csrColInd,
                                               bufferSize));

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseSpruneDense2csrNnz(hipsparseHandle_t         handle,
                                              int                       m,
                                              int                       n,
                                              const float*              A,
                                              int                       lda,
                                              const float*              threshold,
                                              const hipsparseMatDescr_t descr,
                                              int*                      csrRowPtr,
                                              int*                      nnzTotalDevHostPtr,
                                              void*                     buffer)
{
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_sprune_dense2csr_nnz((rocsparse_handle)handle,
                                                             m,
                                                             n,
                                                             A,
                                                             lda,
                                                             threshold,
                                                             (const rocsparse_mat_descr)descr,
                                                             csrRowPtr,
                                                             nnzTotalDevHostPtr,
                                                             buffer));

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseDpruneDense2csrNnz(hipsparseHandle_t         handle,
                                              int                       m,
                                              int                       n,
                                              const double*             A,
                                              int                       lda,
                                              const double*             threshold,
                                              const hipsparseMatDescr_t descr,
                                              int*                      csrRowPtr,
                                              int*                      nnzTotalDevHostPtr,
                                              void*                     buffer)
{
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_dprune_dense2csr_nnz((rocsparse_handle)handle,
                                                             m,
                                                             n,
                                                             A,
                                                             lda,
                                                             threshold,
                                                             (const rocsparse_mat_descr)descr,
                                                             csrRowPtr,
                                                             nnzTotalDevHostPtr,
                                                             buffer));

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseSpruneDense2csr(hipsparseHandle_t         handle,
                                           int                       m,
                                           int                       n,
                                           const float*              A,
                                           int                       lda,
                                           const float*              threshold,
                                           const hipsparseMatDescr_t descr,
                                           float*                    csrVal,
                                           const int*                csrRowPtr,
                                           int*                      csrColInd,
                                           void*                     buffer)
{
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_sprune_dense2csr((rocsparse_handle)handle,
                                                         m,
                                                         n,
                                                         A,
                                                         lda,
                                                         threshold,
                                                         (const rocsparse_mat_descr)descr,
                                                         csrVal,
                                                         csrRowPtr,
                                                         csrColInd,
                                                         buffer));

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseDpruneDense2csr(hipsparseHandle_t         handle,
                                           int                       m,
                                           int                       n,
                                           const double*             A,
                                           int                       lda,
                                           const double*             threshold,
                                           const hipsparseMatDescr_t descr,
                                           double*                   csrVal,
                                           const int*                csrRowPtr,
                                           int*                      csrColInd,
                                           void*                     buffer)
{
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_dprune_dense2csr((rocsparse_handle)handle,
                                                         m,
                                                         n,
                                                         A,
                                                         lda,
                                                         threshold,
                                                         (const rocsparse_mat_descr)descr,
                                                         csrVal,
                                                         csrRowPtr,
                                                         csrColInd,
                                                         buffer));

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseSpruneDense2csrByPercentage_bufferSize(hipsparseHandle_t handle,
                                                                  int               m,
                                                                  int               n,
                                                                  const float*      A,
                                                                  int               lda,
                                                                  float             percentage,
                                                                  const hipsparseMatDescr_t descr,
                                                                  const float*              csrVal,
                                                                  const int*  csrRowPtr,
                                                                  const int*  csrColInd,
                                                                  pruneInfo_t info,
                                                                  size_t*     bufferSize)
{
    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse_sprune_dense2csr_by_percentage_buffer_size((rocsparse_handle)handle,
                                                             m,
                                                             n,
                                                             A,
                                                             lda,
                                                             percentage,
                                                             (const rocsparse_mat_descr)descr,
                                                             csrVal,
                                                             csrRowPtr,
                                                             csrColInd,
                                                             (rocsparse_mat_info)info,
                                                             bufferSize));

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseDpruneDense2csrByPercentage_bufferSize(hipsparseHandle_t handle,
                                                                  int               m,
                                                                  int               n,
                                                                  const double*     A,
                                                                  int               lda,
                                                                  double            percentage,
                                                                  const hipsparseMatDescr_t descr,
                                                                  const double*             csrVal,
                                                                  const int*  csrRowPtr,
                                                                  const int*  csrColInd,
                                                                  pruneInfo_t info,
                                                                  size_t*     bufferSize)
{
    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse_dprune_dense2csr_by_percentage_buffer_size((rocsparse_handle)handle,
                                                             m,
                                                             n,
                                                             A,
                                                             lda,
                                                             percentage,
                                                             (const rocsparse_mat_descr)descr,
                                                             csrVal,
                                                             csrRowPtr,
                                                             csrColInd,
                                                             (rocsparse_mat_info)info,
                                                             bufferSize));

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t
    hipsparseSpruneDense2csrByPercentage_bufferSizeExt(hipsparseHandle_t         handle,
                                                       int                       m,
                                                       int                       n,
                                                       const float*              A,
                                                       int                       lda,
                                                       float                     percentage,
                                                       const hipsparseMatDescr_t descr,
                                                       const float*              csrVal,
                                                       const int*                csrRowPtr,
                                                       const int*                csrColInd,
                                                       pruneInfo_t               info,
                                                       size_t*                   bufferSize)
{
    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse_sprune_dense2csr_by_percentage_buffer_size((rocsparse_handle)handle,
                                                             m,
                                                             n,
                                                             A,
                                                             lda,
                                                             percentage,
                                                             (const rocsparse_mat_descr)descr,
                                                             csrVal,
                                                             csrRowPtr,
                                                             csrColInd,
                                                             (rocsparse_mat_info)info,
                                                             bufferSize));

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t
    hipsparseDpruneDense2csrByPercentage_bufferSizeExt(hipsparseHandle_t         handle,
                                                       int                       m,
                                                       int                       n,
                                                       const double*             A,
                                                       int                       lda,
                                                       double                    percentage,
                                                       const hipsparseMatDescr_t descr,
                                                       const double*             csrVal,
                                                       const int*                csrRowPtr,
                                                       const int*                csrColInd,
                                                       pruneInfo_t               info,
                                                       size_t*                   bufferSize)
{
    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse_dprune_dense2csr_by_percentage_buffer_size((rocsparse_handle)handle,
                                                             m,
                                                             n,
                                                             A,
                                                             lda,
                                                             percentage,
                                                             (const rocsparse_mat_descr)descr,
                                                             csrVal,
                                                             csrRowPtr,
                                                             csrColInd,
                                                             (rocsparse_mat_info)info,
                                                             bufferSize));

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseSpruneDense2csrNnzByPercentage(hipsparseHandle_t         handle,
                                                          int                       m,
                                                          int                       n,
                                                          const float*              A,
                                                          int                       lda,
                                                          float                     percentage,
                                                          const hipsparseMatDescr_t descr,
                                                          int*                      csrRowPtr,
                                                          int*        nnzTotalDevHostPtr,
                                                          pruneInfo_t info,
                                                          void*       buffer)
{
    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse_sprune_dense2csr_nnz_by_percentage((rocsparse_handle)handle,
                                                     m,
                                                     n,
                                                     A,
                                                     lda,
                                                     percentage,
                                                     (const rocsparse_mat_descr)descr,
                                                     csrRowPtr,
                                                     nnzTotalDevHostPtr,
                                                     (rocsparse_mat_info)info,
                                                     buffer));

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseDpruneDense2csrNnzByPercentage(hipsparseHandle_t         handle,
                                                          int                       m,
                                                          int                       n,
                                                          const double*             A,
                                                          int                       lda,
                                                          double                    percentage,
                                                          const hipsparseMatDescr_t descr,
                                                          int*                      csrRowPtr,
                                                          int*        nnzTotalDevHostPtr,
                                                          pruneInfo_t info,
                                                          void*       buffer)
{
    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse_dprune_dense2csr_nnz_by_percentage((rocsparse_handle)handle,
                                                     m,
                                                     n,
                                                     A,
                                                     lda,
                                                     percentage,
                                                     (const rocsparse_mat_descr)descr,
                                                     csrRowPtr,
                                                     nnzTotalDevHostPtr,
                                                     (rocsparse_mat_info)info,
                                                     buffer));

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseSpruneDense2csrByPercentage(hipsparseHandle_t         handle,
                                                       int                       m,
                                                       int                       n,
                                                       const float*              A,
                                                       int                       lda,
                                                       float                     percentage,
                                                       const hipsparseMatDescr_t descr,
                                                       float*                    csrVal,
                                                       const int*                csrRowPtr,
                                                       int*                      csrColInd,
                                                       pruneInfo_t               info,
                                                       void*                     buffer)
{
    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse_sprune_dense2csr_by_percentage((rocsparse_handle)handle,
                                                 m,
                                                 n,
                                                 A,
                                                 lda,
                                                 percentage,
                                                 (const rocsparse_mat_descr)descr,
                                                 csrVal,
                                                 csrRowPtr,
                                                 csrColInd,
                                                 (rocsparse_mat_info)info,
                                                 buffer));

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseDpruneDense2csrByPercentage(hipsparseHandle_t         handle,
                                                       int                       m,
                                                       int                       n,
                                                       const double*             A,
                                                       int                       lda,
                                                       double                    percentage,
                                                       const hipsparseMatDescr_t descr,
                                                       double*                   csrVal,
                                                       const int*                csrRowPtr,
                                                       int*                      csrColInd,
                                                       pruneInfo_t               info,
                                                       void*                     buffer)
{
    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse_dprune_dense2csr_by_percentage((rocsparse_handle)handle,
                                                 m,
                                                 n,
                                                 A,
                                                 lda,
                                                 percentage,
                                                 (const rocsparse_mat_descr)descr,
                                                 csrVal,
                                                 csrRowPtr,
                                                 csrColInd,
                                                 (rocsparse_mat_info)info,
                                                 buffer));

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseSdense2csc(hipsparseHandle_t         handle,
                                      int                       m,
                                      int                       n,
                                      const hipsparseMatDescr_t descr,
                                      const float*              A,
                                      int                       ld,
                                      const int*                nnz_per_columns,
                                      float*                    cscVal,
                                      int*                      cscRowInd,
                                      int*                      cscColPtr)
{
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_sdense2csc((rocsparse_handle)handle,
                                                   m,
                                                   n,
                                                   (const rocsparse_mat_descr)descr,
                                                   A,
                                                   ld,
                                                   nnz_per_columns,
                                                   cscVal,
                                                   cscColPtr,
                                                   cscRowInd));
    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseDdense2csc(hipsparseHandle_t         handle,
                                      int                       m,
                                      int                       n,
                                      const hipsparseMatDescr_t descr,
                                      const double*             A,
                                      int                       ld,
                                      const int*                nnz_per_columns,
                                      double*                   cscVal,
                                      int*                      cscRowInd,
                                      int*                      cscColPtr)
{
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_ddense2csc((rocsparse_handle)handle,
                                                   m,
                                                   n,
                                                   (const rocsparse_mat_descr)descr,
                                                   A,
                                                   ld,
                                                   nnz_per_columns,
                                                   cscVal,
                                                   cscColPtr,
                                                   cscRowInd));
    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseCdense2csc(hipsparseHandle_t         handle,
                                      int                       m,
                                      int                       n,
                                      const hipsparseMatDescr_t descr,
                                      const hipComplex*         A,
                                      int                       ld,
                                      const int*                nnz_per_columns,
                                      hipComplex*               cscVal,
                                      int*                      cscRowInd,
                                      int*                      cscColPtr)
{
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_cdense2csc((rocsparse_handle)handle,
                                                   m,
                                                   n,
                                                   (const rocsparse_mat_descr)descr,
                                                   (const rocsparse_float_complex*)A,
                                                   ld,
                                                   nnz_per_columns,
                                                   (rocsparse_float_complex*)cscVal,
                                                   cscColPtr,
                                                   cscRowInd));
    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseZdense2csc(hipsparseHandle_t         handle,
                                      int                       m,
                                      int                       n,
                                      const hipsparseMatDescr_t descr,
                                      const hipDoubleComplex*   A,
                                      int                       ld,
                                      const int*                nnz_per_columns,
                                      hipDoubleComplex*         cscVal,
                                      int*                      cscRowInd,
                                      int*                      cscColPtr)
{
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_zdense2csc((rocsparse_handle)handle,
                                                   m,
                                                   n,
                                                   (const rocsparse_mat_descr)descr,
                                                   (const rocsparse_double_complex*)A,
                                                   ld,
                                                   nnz_per_columns,
                                                   (rocsparse_double_complex*)cscVal,
                                                   cscColPtr,
                                                   cscRowInd));
    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseScsr2dense(hipsparseHandle_t         handle,
                                      int                       m,
                                      int                       n,
                                      const hipsparseMatDescr_t descr,
                                      const float*              csrVal,
                                      const int*                csrRowPtr,
                                      const int*                csrColInd,
                                      float*                    A,
                                      int                       ld)
{
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_scsr2dense((rocsparse_handle)handle,
                                                   m,
                                                   n,
                                                   (const rocsparse_mat_descr)descr,
                                                   csrVal,
                                                   csrRowPtr,
                                                   csrColInd,
                                                   A,
                                                   ld));
    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseDcsr2dense(hipsparseHandle_t         handle,
                                      int                       m,
                                      int                       n,
                                      const hipsparseMatDescr_t descr,
                                      const double*             csrVal,
                                      const int*                csrRowPtr,
                                      const int*                csrColInd,
                                      double*                   A,
                                      int                       ld)
{
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_dcsr2dense((rocsparse_handle)handle,
                                                   m,
                                                   n,
                                                   (const rocsparse_mat_descr)descr,
                                                   csrVal,
                                                   csrRowPtr,
                                                   csrColInd,
                                                   A,
                                                   ld));
    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseCcsr2dense(hipsparseHandle_t         handle,
                                      int                       m,
                                      int                       n,
                                      const hipsparseMatDescr_t descr,
                                      const hipComplex*         csrVal,
                                      const int*                csrRowPtr,
                                      const int*                csrColInd,
                                      hipComplex*               A,
                                      int                       ld)
{
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_ccsr2dense((rocsparse_handle)handle,
                                                   m,
                                                   n,
                                                   (const rocsparse_mat_descr)descr,
                                                   (const rocsparse_float_complex*)csrVal,
                                                   csrRowPtr,
                                                   csrColInd,
                                                   (rocsparse_float_complex*)A,
                                                   ld));
    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseZcsr2dense(hipsparseHandle_t         handle,
                                      int                       m,
                                      int                       n,
                                      const hipsparseMatDescr_t descr,
                                      const hipDoubleComplex*   csrVal,
                                      const int*                csrRowPtr,
                                      const int*                csrColInd,
                                      hipDoubleComplex*         A,
                                      int                       ld)
{
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_zcsr2dense((rocsparse_handle)handle,
                                                   m,
                                                   n,
                                                   (const rocsparse_mat_descr)descr,
                                                   (const rocsparse_double_complex*)csrVal,
                                                   csrRowPtr,
                                                   csrColInd,
                                                   (rocsparse_double_complex*)A,
                                                   ld));
    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseScsc2dense(hipsparseHandle_t         handle,
                                      int                       m,
                                      int                       n,
                                      const hipsparseMatDescr_t descr,
                                      const float*              cscVal,
                                      const int*                cscRowInd,
                                      const int*                cscColPtr,
                                      float*                    A,
                                      int                       ld)
{
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_scsc2dense((rocsparse_handle)handle,
                                                   m,
                                                   n,
                                                   (const rocsparse_mat_descr)descr,
                                                   cscVal,
                                                   cscColPtr,
                                                   cscRowInd,
                                                   A,
                                                   ld));
    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseDcsc2dense(hipsparseHandle_t         handle,
                                      int                       m,
                                      int                       n,
                                      const hipsparseMatDescr_t descr,
                                      const double*             cscVal,
                                      const int*                cscRowInd,
                                      const int*                cscColPtr,
                                      double*                   A,
                                      int                       ld)
{
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_dcsc2dense((rocsparse_handle)handle,
                                                   m,
                                                   n,
                                                   (const rocsparse_mat_descr)descr,
                                                   cscVal,
                                                   cscColPtr,
                                                   cscRowInd,
                                                   A,
                                                   ld));
    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseCcsc2dense(hipsparseHandle_t         handle,
                                      int                       m,
                                      int                       n,
                                      const hipsparseMatDescr_t descr,
                                      const hipComplex*         cscVal,
                                      const int*                cscRowInd,
                                      const int*                cscColPtr,
                                      hipComplex*               A,
                                      int                       ld)
{
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_ccsc2dense((rocsparse_handle)handle,
                                                   m,
                                                   n,
                                                   (const rocsparse_mat_descr)descr,
                                                   (const rocsparse_float_complex*)cscVal,
                                                   cscColPtr,
                                                   cscRowInd,
                                                   (rocsparse_float_complex*)A,
                                                   ld));
    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseZcsc2dense(hipsparseHandle_t         handle,
                                      int                       m,
                                      int                       n,
                                      const hipsparseMatDescr_t descr,
                                      const hipDoubleComplex*   cscVal,
                                      const int*                cscRowInd,
                                      const int*                cscColPtr,
                                      hipDoubleComplex*         A,
                                      int                       ld)
{
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_zcsc2dense((rocsparse_handle)handle,
                                                   m,
                                                   n,
                                                   (const rocsparse_mat_descr)descr,
                                                   (const rocsparse_double_complex*)cscVal,
                                                   cscColPtr,
                                                   cscRowInd,
                                                   (rocsparse_double_complex*)A,
                                                   ld));
    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseXcsr2bsrNnz(hipsparseHandle_t         handle,
                                       hipsparseDirection_t      dirA,
                                       int                       m,
                                       int                       n,
                                       const hipsparseMatDescr_t descrA,
                                       const int*                csrRowPtrA,
                                       const int*                csrColIndA,
                                       int                       blockDim,
                                       const hipsparseMatDescr_t descrC,
                                       int*                      bsrRowPtrC,
                                       int*                      bsrNnzb)
{
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_csr2bsr_nnz((rocsparse_handle)handle,
                                                    hipDirectionToHCCDirection(dirA),
                                                    m,
                                                    n,
                                                    (const rocsparse_mat_descr)descrA,
                                                    csrRowPtrA,
                                                    csrColIndA,
                                                    blockDim,
                                                    (const rocsparse_mat_descr)descrC,
                                                    bsrRowPtrC,
                                                    bsrNnzb));

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseSnnz_compress(hipsparseHandle_t         handle,
                                         int                       m,
                                         const hipsparseMatDescr_t descrA,
                                         const float*              csrValA,
                                         const int*                csrRowPtrA,
                                         int*                      nnzPerRow,
                                         int*                      nnzC,
                                         float                     tol)
{
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_snnz_compress((rocsparse_handle)handle,
                                                      m,
                                                      (const rocsparse_mat_descr)descrA,
                                                      csrValA,
                                                      csrRowPtrA,
                                                      nnzPerRow,
                                                      nnzC,
                                                      tol));
    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseDnnz_compress(hipsparseHandle_t         handle,
                                         int                       m,
                                         const hipsparseMatDescr_t descrA,
                                         const double*             csrValA,
                                         const int*                csrRowPtrA,
                                         int*                      nnzPerRow,
                                         int*                      nnzC,
                                         double                    tol)
{
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_dnnz_compress((rocsparse_handle)handle,
                                                      m,
                                                      (const rocsparse_mat_descr)descrA,
                                                      csrValA,
                                                      csrRowPtrA,
                                                      nnzPerRow,
                                                      nnzC,
                                                      tol));
    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseCnnz_compress(hipsparseHandle_t         handle,
                                         int                       m,
                                         const hipsparseMatDescr_t descrA,
                                         const hipComplex*         csrValA,
                                         const int*                csrRowPtrA,
                                         int*                      nnzPerRow,
                                         int*                      nnzC,
                                         hipComplex                tol)
{
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_cnnz_compress((rocsparse_handle)handle,
                                                      m,
                                                      (const rocsparse_mat_descr)descrA,
                                                      (const rocsparse_float_complex*)csrValA,
                                                      csrRowPtrA,
                                                      nnzPerRow,
                                                      nnzC,
                                                      {hipCrealf(tol), hipCimagf(tol)}));
    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseZnnz_compress(hipsparseHandle_t         handle,
                                         int                       m,
                                         const hipsparseMatDescr_t descrA,
                                         const hipDoubleComplex*   csrValA,
                                         const int*                csrRowPtrA,
                                         int*                      nnzPerRow,
                                         int*                      nnzC,
                                         hipDoubleComplex          tol)
{
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_znnz_compress((rocsparse_handle)handle,
                                                      m,
                                                      (const rocsparse_mat_descr)descrA,
                                                      (const rocsparse_double_complex*)csrValA,
                                                      csrRowPtrA,
                                                      nnzPerRow,
                                                      nnzC,
                                                      {hipCreal(tol), hipCimag(tol)}));
    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseXcoo2csr(hipsparseHandle_t    handle,
                                    const int*           cooRowInd,
                                    int                  nnz,
                                    int                  m,
                                    int*                 csrRowPtr,
                                    hipsparseIndexBase_t idxBase)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_coo2csr(
        (rocsparse_handle)handle, cooRowInd, nnz, m, csrRowPtr, hipBaseToHCCBase(idxBase)));
}

hipsparseStatus_t hipsparseCreateIdentityPermutation(hipsparseHandle_t handle, int n, int* p)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_create_identity_permutation((rocsparse_handle)handle, n, p));
}

hipsparseStatus_t hipsparseXcsrsort_bufferSizeExt(hipsparseHandle_t handle,
                                                  int               m,
                                                  int               n,
                                                  int               nnz,
                                                  const int*        csrRowPtr,
                                                  const int*        csrColInd,
                                                  size_t*           pBufferSizeInBytes)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_csrsort_buffer_size(
        (rocsparse_handle)handle, m, n, nnz, csrRowPtr, csrColInd, pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseXcsrsort(hipsparseHandle_t         handle,
                                    int                       m,
                                    int                       n,
                                    int                       nnz,
                                    const hipsparseMatDescr_t descrA,
                                    const int*                csrRowPtr,
                                    int*                      csrColInd,
                                    int*                      P,
                                    void*                     pBuffer)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_csrsort((rocsparse_handle)handle,
                                                        m,
                                                        n,
                                                        nnz,
                                                        (rocsparse_mat_descr)descrA,
                                                        csrRowPtr,
                                                        csrColInd,
                                                        P,
                                                        pBuffer));
}

hipsparseStatus_t hipsparseXcscsort_bufferSizeExt(hipsparseHandle_t handle,
                                                  int               m,
                                                  int               n,
                                                  int               nnz,
                                                  const int*        cscColPtr,
                                                  const int*        cscRowInd,
                                                  size_t*           pBufferSizeInBytes)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_cscsort_buffer_size(
        (rocsparse_handle)handle, m, n, nnz, cscColPtr, cscRowInd, pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseXcscsort(hipsparseHandle_t         handle,
                                    int                       m,
                                    int                       n,
                                    int                       nnz,
                                    const hipsparseMatDescr_t descrA,
                                    const int*                cscColPtr,
                                    int*                      cscRowInd,
                                    int*                      P,
                                    void*                     pBuffer)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_cscsort((rocsparse_handle)handle,
                                                        m,
                                                        n,
                                                        nnz,
                                                        (rocsparse_mat_descr)descrA,
                                                        cscColPtr,
                                                        cscRowInd,
                                                        P,
                                                        pBuffer));
}

hipsparseStatus_t hipsparseXcoosort_bufferSizeExt(hipsparseHandle_t handle,
                                                  int               m,
                                                  int               n,
                                                  int               nnz,
                                                  const int*        cooRows,
                                                  const int*        cooCols,
                                                  size_t*           pBufferSizeInBytes)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_coosort_buffer_size(
        (rocsparse_handle)handle, m, n, nnz, cooRows, cooCols, pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseXcoosortByRow(hipsparseHandle_t handle,
                                         int               m,
                                         int               n,
                                         int               nnz,
                                         int*              cooRows,
                                         int*              cooCols,
                                         int*              P,
                                         void*             pBuffer)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_coosort_by_row(
        (rocsparse_handle)handle, m, n, nnz, cooRows, cooCols, P, pBuffer));
}

hipsparseStatus_t hipsparseXcoosortByColumn(hipsparseHandle_t handle,
                                            int               m,
                                            int               n,
                                            int               nnz,
                                            int*              cooRows,
                                            int*              cooCols,
                                            int*              P,
                                            void*             pBuffer)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_coosort_by_column(
        (rocsparse_handle)handle, m, n, nnz, cooRows, cooCols, P, pBuffer));
}

hipsparseStatus_t hipsparseSgebsr2gebsr_bufferSize(hipsparseHandle_t         handle,
                                                   hipsparseDirection_t      dirA,
                                                   int                       mb,
                                                   int                       nb,
                                                   int                       nnzb,
                                                   const hipsparseMatDescr_t descrA,
                                                   const float*              bsrValA,
                                                   const int*                bsrRowPtrA,
                                                   const int*                bsrColIndA,
                                                   int                       rowBlockDimA,
                                                   int                       colBlockDimA,
                                                   int                       rowBlockDimC,
                                                   int                       colBlockDimC,
                                                   int*                      bufferSize)
{
    size_t bufSize;
    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse_sgebsr2gebsr_buffer_size((rocsparse_handle)handle,
                                           hipDirectionToHCCDirection(dirA),
                                           mb,
                                           nb,
                                           nnzb,
                                           (const rocsparse_mat_descr)descrA,
                                           bsrValA,
                                           bsrRowPtrA,
                                           bsrColIndA,
                                           rowBlockDimA,
                                           colBlockDimA,
                                           rowBlockDimC,
                                           colBlockDimC,
                                           bufferSize != nullptr ? &bufSize : nullptr));

    if(bufferSize != nullptr)
    {
        *bufferSize = bufSize;
    }
    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseDgebsr2gebsr_bufferSize(hipsparseHandle_t         handle,
                                                   hipsparseDirection_t      dirA,
                                                   int                       mb,
                                                   int                       nb,
                                                   int                       nnzb,
                                                   const hipsparseMatDescr_t descrA,
                                                   const double*             bsrValA,
                                                   const int*                bsrRowPtrA,
                                                   const int*                bsrColIndA,
                                                   int                       rowBlockDimA,
                                                   int                       colBlockDimA,
                                                   int                       rowBlockDimC,
                                                   int                       colBlockDimC,
                                                   int*                      bufferSize)
{
    size_t bufSize;
    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse_dgebsr2gebsr_buffer_size((rocsparse_handle)handle,
                                           hipDirectionToHCCDirection(dirA),
                                           mb,
                                           nb,
                                           nnzb,
                                           (const rocsparse_mat_descr)descrA,
                                           bsrValA,
                                           bsrRowPtrA,
                                           bsrColIndA,
                                           rowBlockDimA,
                                           colBlockDimA,
                                           rowBlockDimC,
                                           colBlockDimC,
                                           bufferSize != nullptr ? &bufSize : nullptr));
    if(bufferSize != nullptr)
    {
        *bufferSize = bufSize;
    }
    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseCgebsr2gebsr_bufferSize(hipsparseHandle_t         handle,
                                                   hipsparseDirection_t      dirA,
                                                   int                       mb,
                                                   int                       nb,
                                                   int                       nnzb,
                                                   const hipsparseMatDescr_t descrA,
                                                   const hipComplex*         bsrValA,
                                                   const int*                bsrRowPtrA,
                                                   const int*                bsrColIndA,
                                                   int                       rowBlockDimA,
                                                   int                       colBlockDimA,
                                                   int                       rowBlockDimC,
                                                   int                       colBlockDimC,
                                                   int*                      bufferSize)
{
    size_t bufSize;
    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse_cgebsr2gebsr_buffer_size((rocsparse_handle)handle,
                                           hipDirectionToHCCDirection(dirA),
                                           mb,
                                           nb,
                                           nnzb,
                                           (const rocsparse_mat_descr)descrA,
                                           (const rocsparse_float_complex*)bsrValA,
                                           bsrRowPtrA,
                                           bsrColIndA,
                                           rowBlockDimA,
                                           colBlockDimA,
                                           rowBlockDimC,
                                           colBlockDimC,
                                           bufferSize != nullptr ? &bufSize : nullptr));
    if(bufferSize != nullptr)
    {
        *bufferSize = bufSize;
    }
    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseZgebsr2gebsr_bufferSize(hipsparseHandle_t         handle,
                                                   hipsparseDirection_t      dirA,
                                                   int                       mb,
                                                   int                       nb,
                                                   int                       nnzb,
                                                   const hipsparseMatDescr_t descrA,
                                                   const hipDoubleComplex*   bsrValA,
                                                   const int*                bsrRowPtrA,
                                                   const int*                bsrColIndA,
                                                   int                       rowBlockDimA,
                                                   int                       colBlockDimA,
                                                   int                       rowBlockDimC,
                                                   int                       colBlockDimC,
                                                   int*                      bufferSize)
{
    size_t bufSize;
    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse_zgebsr2gebsr_buffer_size((rocsparse_handle)handle,
                                           hipDirectionToHCCDirection(dirA),
                                           mb,
                                           nb,
                                           nnzb,
                                           (const rocsparse_mat_descr)descrA,
                                           (const rocsparse_double_complex*)bsrValA,
                                           bsrRowPtrA,
                                           bsrColIndA,
                                           rowBlockDimA,
                                           colBlockDimA,
                                           rowBlockDimC,
                                           colBlockDimC,
                                           bufferSize != nullptr ? &bufSize : nullptr));
    if(bufferSize != nullptr)
    {
        *bufferSize = bufSize;
    }
    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseXgebsr2gebsrNnz(hipsparseHandle_t         handle,
                                           hipsparseDirection_t      dirA,
                                           int                       mb,
                                           int                       nb,
                                           int                       nnzb,
                                           const hipsparseMatDescr_t descrA,
                                           const int*                bsrRowPtrA,
                                           const int*                bsrColIndA,
                                           int                       rowBlockDimA,
                                           int                       colBlockDimA,
                                           const hipsparseMatDescr_t descrC,
                                           int*                      bsrRowPtrC,
                                           int                       rowBlockDimC,
                                           int                       colBlockDimC,
                                           int*                      nnzTotalDevHostPtr,
                                           void*                     buffer)
{
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_gebsr2gebsr_nnz((rocsparse_handle)handle,
                                                        hipDirectionToHCCDirection(dirA),
                                                        mb,
                                                        nb,
                                                        nnzb,
                                                        (const rocsparse_mat_descr)descrA,
                                                        bsrRowPtrA,
                                                        bsrColIndA,
                                                        rowBlockDimA,
                                                        colBlockDimA,
                                                        (const rocsparse_mat_descr)descrC,
                                                        bsrRowPtrC,
                                                        rowBlockDimC,
                                                        colBlockDimC,
                                                        nnzTotalDevHostPtr,
                                                        buffer));
    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseSgebsr2gebsr(hipsparseHandle_t         handle,
                                        hipsparseDirection_t      dirA,
                                        int                       mb,
                                        int                       nb,
                                        int                       nnzb,
                                        const hipsparseMatDescr_t descrA,
                                        const float*              bsrValA,
                                        const int*                bsrRowPtrA,
                                        const int*                bsrColIndA,
                                        int                       rowBlockDimA,
                                        int                       colBlockDimA,
                                        const hipsparseMatDescr_t descrC,
                                        float*                    bsrValC,
                                        int*                      bsrRowPtrC,
                                        int*                      bsrColIndC,
                                        int                       rowBlockDimC,
                                        int                       colBlockDimC,
                                        void*                     buffer)
{
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_sgebsr2gebsr((rocsparse_handle)handle,
                                                     hipDirectionToHCCDirection(dirA),
                                                     mb,
                                                     nb,
                                                     nnzb,
                                                     (const rocsparse_mat_descr)descrA,
                                                     bsrValA,
                                                     bsrRowPtrA,
                                                     bsrColIndA,
                                                     rowBlockDimA,
                                                     colBlockDimA,
                                                     (const rocsparse_mat_descr)descrC,
                                                     bsrValC,
                                                     bsrRowPtrC,
                                                     bsrColIndC,
                                                     rowBlockDimC,
                                                     colBlockDimC,
                                                     buffer));
    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseDgebsr2gebsr(hipsparseHandle_t         handle,
                                        hipsparseDirection_t      dirA,
                                        int                       mb,
                                        int                       nb,
                                        int                       nnzb,
                                        const hipsparseMatDescr_t descrA,
                                        const double*             bsrValA,
                                        const int*                bsrRowPtrA,
                                        const int*                bsrColIndA,
                                        int                       rowBlockDimA,
                                        int                       colBlockDimA,
                                        const hipsparseMatDescr_t descrC,
                                        double*                   bsrValC,
                                        int*                      bsrRowPtrC,
                                        int*                      bsrColIndC,
                                        int                       rowBlockDimC,
                                        int                       colBlockDimC,
                                        void*                     buffer)
{
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_dgebsr2gebsr((rocsparse_handle)handle,
                                                     hipDirectionToHCCDirection(dirA),
                                                     mb,
                                                     nb,
                                                     nnzb,
                                                     (const rocsparse_mat_descr)descrA,
                                                     bsrValA,
                                                     bsrRowPtrA,
                                                     bsrColIndA,
                                                     rowBlockDimA,
                                                     colBlockDimA,
                                                     (const rocsparse_mat_descr)descrC,
                                                     bsrValC,
                                                     bsrRowPtrC,
                                                     bsrColIndC,
                                                     rowBlockDimC,
                                                     colBlockDimC,
                                                     buffer));
    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseCgebsr2gebsr(hipsparseHandle_t         handle,
                                        hipsparseDirection_t      dirA,
                                        int                       mb,
                                        int                       nb,
                                        int                       nnzb,
                                        const hipsparseMatDescr_t descrA,
                                        const hipComplex*         bsrValA,
                                        const int*                bsrRowPtrA,
                                        const int*                bsrColIndA,
                                        int                       rowBlockDimA,
                                        int                       colBlockDimA,
                                        const hipsparseMatDescr_t descrC,
                                        hipComplex*               bsrValC,
                                        int*                      bsrRowPtrC,
                                        int*                      bsrColIndC,
                                        int                       rowBlockDimC,
                                        int                       colBlockDimC,
                                        void*                     buffer)
{
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_cgebsr2gebsr((rocsparse_handle)handle,
                                                     hipDirectionToHCCDirection(dirA),
                                                     mb,
                                                     nb,
                                                     nnzb,
                                                     (const rocsparse_mat_descr)descrA,
                                                     (const rocsparse_float_complex*)bsrValA,
                                                     bsrRowPtrA,
                                                     bsrColIndA,
                                                     rowBlockDimA,
                                                     colBlockDimA,
                                                     (const rocsparse_mat_descr)descrC,
                                                     (rocsparse_float_complex*)bsrValC,
                                                     bsrRowPtrC,
                                                     bsrColIndC,
                                                     rowBlockDimC,
                                                     colBlockDimC,
                                                     buffer));
    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseZgebsr2gebsr(hipsparseHandle_t         handle,
                                        hipsparseDirection_t      dirA,
                                        int                       mb,
                                        int                       nb,
                                        int                       nnzb,
                                        const hipsparseMatDescr_t descrA,
                                        const hipDoubleComplex*   bsrValA,
                                        const int*                bsrRowPtrA,
                                        const int*                bsrColIndA,
                                        int                       rowBlockDimA,
                                        int                       colBlockDimA,
                                        const hipsparseMatDescr_t descrC,
                                        hipDoubleComplex*         bsrValC,
                                        int*                      bsrRowPtrC,
                                        int*                      bsrColIndC,
                                        int                       rowBlockDimC,
                                        int                       colBlockDimC,
                                        void*                     buffer)
{
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_zgebsr2gebsr((rocsparse_handle)handle,
                                                     hipDirectionToHCCDirection(dirA),
                                                     mb,
                                                     nb,
                                                     nnzb,
                                                     (const rocsparse_mat_descr)descrA,
                                                     (const rocsparse_double_complex*)bsrValA,
                                                     bsrRowPtrA,
                                                     bsrColIndA,
                                                     rowBlockDimA,
                                                     colBlockDimA,
                                                     (const rocsparse_mat_descr)descrC,
                                                     (rocsparse_double_complex*)bsrValC,
                                                     bsrRowPtrC,
                                                     bsrColIndC,
                                                     rowBlockDimC,
                                                     colBlockDimC,
                                                     buffer));
    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseScsru2csr_bufferSizeExt(hipsparseHandle_t handle,
                                                   int               m,
                                                   int               n,
                                                   int               nnz,
                                                   float*            csrVal,
                                                   const int*        csrRowPtr,
                                                   int*              csrColInd,
                                                   csru2csrInfo_t    info,
                                                   size_t*           pBufferSizeInBytes)
{
    // Test for bad args
    if(handle == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    // Invalid sizes
    if(m < 0 || n < 0 || nnz < 0)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    // Quick return
    if(m == 0 || n == 0 || nnz == 0)
    {
        // nnz must be 0 and pBufferSizeInBytes must be valid
        if(nnz != 0 || pBufferSizeInBytes == nullptr)
        {
            return HIPSPARSE_STATUS_INVALID_VALUE;
        }

        *pBufferSizeInBytes = 4;

        return HIPSPARSE_STATUS_SUCCESS;
    }

    // Invalid pointers
    if(csrVal == nullptr || csrRowPtr == nullptr || csrColInd == nullptr || info == nullptr
       || pBufferSizeInBytes == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    // Determine required buffer size for CSR sort
    RETURN_IF_HIPSPARSE_ERROR(hipsparseXcsrsort_bufferSizeExt(
        handle, m, n, nnz, csrRowPtr, csrColInd, pBufferSizeInBytes));

    // We need a buffer of at least nnz * sizeof(float)
    size_t min_size     = nnz * sizeof(float);
    *pBufferSizeInBytes = std::max(*pBufferSizeInBytes, min_size);

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseDcsru2csr_bufferSizeExt(hipsparseHandle_t handle,
                                                   int               m,
                                                   int               n,
                                                   int               nnz,
                                                   double*           csrVal,
                                                   const int*        csrRowPtr,
                                                   int*              csrColInd,
                                                   csru2csrInfo_t    info,
                                                   size_t*           pBufferSizeInBytes)
{
    // Test for bad args
    if(handle == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    // Invalid sizes
    if(m < 0 || n < 0 || nnz < 0)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    // Quick return
    if(m == 0 || n == 0 || nnz == 0)
    {
        // nnz must be 0 and pBufferSizeInBytes must be valid
        if(nnz != 0 || pBufferSizeInBytes == nullptr)
        {
            return HIPSPARSE_STATUS_INVALID_VALUE;
        }

        *pBufferSizeInBytes = 4;

        return HIPSPARSE_STATUS_SUCCESS;
    }

    // Invalid pointers
    if(csrVal == nullptr || csrRowPtr == nullptr || csrColInd == nullptr || info == nullptr
       || pBufferSizeInBytes == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    // Determine required buffer size for CSR sort
    RETURN_IF_HIPSPARSE_ERROR(hipsparseXcsrsort_bufferSizeExt(
        handle, m, n, nnz, csrRowPtr, csrColInd, pBufferSizeInBytes));

    // We need a buffer of at least nnz * sizeof(double)
    size_t min_size     = nnz * sizeof(double);
    *pBufferSizeInBytes = std::max(*pBufferSizeInBytes, min_size);

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseCcsru2csr_bufferSizeExt(hipsparseHandle_t handle,
                                                   int               m,
                                                   int               n,
                                                   int               nnz,
                                                   hipComplex*       csrVal,
                                                   const int*        csrRowPtr,
                                                   int*              csrColInd,
                                                   csru2csrInfo_t    info,
                                                   size_t*           pBufferSizeInBytes)
{
    // Test for bad args
    if(handle == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    // Invalid sizes
    if(m < 0 || n < 0 || nnz < 0)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    // Quick return
    if(m == 0 || n == 0 || nnz == 0)
    {
        // nnz must be 0 and pBufferSizeInBytes must be valid
        if(nnz != 0 || pBufferSizeInBytes == nullptr)
        {
            return HIPSPARSE_STATUS_INVALID_VALUE;
        }

        *pBufferSizeInBytes = 4;

        return HIPSPARSE_STATUS_SUCCESS;
    }

    // Invalid pointers
    if(csrVal == nullptr || csrRowPtr == nullptr || csrColInd == nullptr || info == nullptr
       || pBufferSizeInBytes == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    // Determine required buffer size for CSR sort
    RETURN_IF_HIPSPARSE_ERROR(hipsparseXcsrsort_bufferSizeExt(
        handle, m, n, nnz, csrRowPtr, csrColInd, pBufferSizeInBytes));

    // We need a buffer of at least nnz * sizeof(hipComplex)
    size_t min_size     = nnz * sizeof(hipComplex);
    *pBufferSizeInBytes = std::max(*pBufferSizeInBytes, min_size);

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseZcsru2csr_bufferSizeExt(hipsparseHandle_t handle,
                                                   int               m,
                                                   int               n,
                                                   int               nnz,
                                                   hipDoubleComplex* csrVal,
                                                   const int*        csrRowPtr,
                                                   int*              csrColInd,
                                                   csru2csrInfo_t    info,
                                                   size_t*           pBufferSizeInBytes)
{
    // Test for bad args
    if(handle == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    // Invalid sizes
    if(m < 0 || n < 0 || nnz < 0)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    // Quick return
    if(m == 0 || n == 0 || nnz == 0)
    {
        // nnz must be 0 and pBufferSizeInBytes must be valid
        if(nnz != 0 || pBufferSizeInBytes == nullptr)
        {
            return HIPSPARSE_STATUS_INVALID_VALUE;
        }

        *pBufferSizeInBytes = 4;

        return HIPSPARSE_STATUS_SUCCESS;
    }

    // Invalid pointers
    if(csrVal == nullptr || csrRowPtr == nullptr || csrColInd == nullptr || info == nullptr
       || pBufferSizeInBytes == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    // Determine required buffer size for CSR sort
    RETURN_IF_HIPSPARSE_ERROR(hipsparseXcsrsort_bufferSizeExt(
        handle, m, n, nnz, csrRowPtr, csrColInd, pBufferSizeInBytes));

    // We need a buffer of at least nnz * sizeof(hipDoubleComplex)
    size_t min_size     = nnz * sizeof(hipDoubleComplex);
    *pBufferSizeInBytes = std::max(*pBufferSizeInBytes, min_size);

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseScsru2csr(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       n,
                                     int                       nnz,
                                     const hipsparseMatDescr_t descrA,
                                     float*                    csrVal,
                                     const int*                csrRowPtr,
                                     int*                      csrColInd,
                                     csru2csrInfo_t            info,
                                     void*                     pBuffer)
{
    // Test for bad args
    if(handle == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    // Invalid sizes
    if(m < 0 || n < 0 || nnz < 0)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    // Quick return
    if(m == 0 || n == 0 || nnz == 0)
    {
        // nnz must be 0
        if(nnz != 0)
        {
            return HIPSPARSE_STATUS_INVALID_VALUE;
        }

        return HIPSPARSE_STATUS_SUCCESS;
    }

    // Invalid pointers
    if(descrA == nullptr || csrVal == nullptr || csrRowPtr == nullptr || csrColInd == nullptr
       || info == nullptr || pBuffer == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    // De-allocate permutation array, if already allocated but sizes do not match
    if(info->P != nullptr && info->size != nnz)
    {
        RETURN_IF_HIP_ERROR(hipFree(info->P));
        info->size = 0;
    }

    // Allocate memory inside info structure to keep track of the permutation
    // if it has not yet been allocated with matching size
    if(info->P == nullptr)
    {
        // size must be 0
        assert(info->size == 0);

        RETURN_IF_HIP_ERROR(hipMalloc((void**)&info->P, sizeof(int) * nnz));

        info->size = nnz;
    }

    // Initialize permutation with identity
    RETURN_IF_HIPSPARSE_ERROR(hipsparseCreateIdentityPermutation(handle, nnz, info->P));

    // Sort CSR columns
    RETURN_IF_HIPSPARSE_ERROR(
        hipsparseXcsrsort(handle, m, n, nnz, descrA, csrRowPtr, csrColInd, info->P, pBuffer));

    // Sort CSR values
    RETURN_IF_HIPSPARSE_ERROR(
        hipsparseSgthr(handle, nnz, csrVal, (float*)pBuffer, info->P, HIPSPARSE_INDEX_BASE_ZERO));

    // Get stream
    hipStream_t stream;
    RETURN_IF_HIPSPARSE_ERROR(hipsparseGetStream(handle, &stream));

    // Copy sorted values back to csrVal
    RETURN_IF_HIP_ERROR(
        hipMemcpyAsync(csrVal, pBuffer, sizeof(float) * nnz, hipMemcpyDeviceToDevice, stream));

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseDcsru2csr(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       n,
                                     int                       nnz,
                                     const hipsparseMatDescr_t descrA,
                                     double*                   csrVal,
                                     const int*                csrRowPtr,
                                     int*                      csrColInd,
                                     csru2csrInfo_t            info,
                                     void*                     pBuffer)
{
    // Test for bad args
    if(handle == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    // Invalid sizes
    if(m < 0 || n < 0 || nnz < 0)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    // Quick return
    if(m == 0 || n == 0 || nnz == 0)
    {
        // nnz must be 0
        if(nnz != 0)
        {
            return HIPSPARSE_STATUS_INVALID_VALUE;
        }

        return HIPSPARSE_STATUS_SUCCESS;
    }

    // Invalid pointers
    if(descrA == nullptr || csrVal == nullptr || csrRowPtr == nullptr || csrColInd == nullptr
       || info == nullptr || pBuffer == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    // De-allocate permutation array, if already allocated but sizes do not match
    if(info->P != nullptr && info->size != nnz)
    {
        RETURN_IF_HIP_ERROR(hipFree(info->P));
        info->size = 0;
    }

    // Allocate memory inside info structure to keep track of the permutation
    // if it has not yet been allocated with matching size
    if(info->P == nullptr)
    {
        // size must be 0
        assert(info->size == 0);

        RETURN_IF_HIP_ERROR(hipMalloc((void**)&info->P, sizeof(int) * nnz));

        info->size = nnz;
    }

    // Initialize permutation with identity
    RETURN_IF_HIPSPARSE_ERROR(hipsparseCreateIdentityPermutation(handle, nnz, info->P));

    // Sort CSR columns
    RETURN_IF_HIPSPARSE_ERROR(
        hipsparseXcsrsort(handle, m, n, nnz, descrA, csrRowPtr, csrColInd, info->P, pBuffer));

    // Sort CSR values
    RETURN_IF_HIPSPARSE_ERROR(
        hipsparseDgthr(handle, nnz, csrVal, (double*)pBuffer, info->P, HIPSPARSE_INDEX_BASE_ZERO));

    // Get stream
    hipStream_t stream;
    RETURN_IF_HIPSPARSE_ERROR(hipsparseGetStream(handle, &stream));

    // Copy sorted values back to csrVal
    RETURN_IF_HIP_ERROR(
        hipMemcpyAsync(csrVal, pBuffer, sizeof(double) * nnz, hipMemcpyDeviceToDevice, stream));

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseCcsru2csr(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       n,
                                     int                       nnz,
                                     const hipsparseMatDescr_t descrA,
                                     hipComplex*               csrVal,
                                     const int*                csrRowPtr,
                                     int*                      csrColInd,
                                     csru2csrInfo_t            info,
                                     void*                     pBuffer)
{
    // Test for bad args
    if(handle == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    // Invalid sizes
    if(m < 0 || n < 0 || nnz < 0)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    // Quick return
    if(m == 0 || n == 0 || nnz == 0)
    {
        // nnz must be 0
        if(nnz != 0)
        {
            return HIPSPARSE_STATUS_INVALID_VALUE;
        }

        return HIPSPARSE_STATUS_SUCCESS;
    }

    // Invalid pointers
    if(descrA == nullptr || csrVal == nullptr || csrRowPtr == nullptr || csrColInd == nullptr
       || info == nullptr || pBuffer == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    // De-allocate permutation array, if already allocated but sizes do not match
    if(info->P != nullptr && info->size != nnz)
    {
        RETURN_IF_HIP_ERROR(hipFree(info->P));
        info->size = 0;
    }

    // Allocate memory inside info structure to keep track of the permutation
    // if it has not yet been allocated with matching size
    if(info->P == nullptr)
    {
        // size must be 0
        assert(info->size == 0);

        RETURN_IF_HIP_ERROR(hipMalloc((void**)&info->P, sizeof(int) * nnz));

        info->size = nnz;
    }

    // Initialize permutation with identity
    RETURN_IF_HIPSPARSE_ERROR(hipsparseCreateIdentityPermutation(handle, nnz, info->P));

    // Sort CSR columns
    RETURN_IF_HIPSPARSE_ERROR(
        hipsparseXcsrsort(handle, m, n, nnz, descrA, csrRowPtr, csrColInd, info->P, pBuffer));

    // Sort CSR values
    RETURN_IF_HIPSPARSE_ERROR(hipsparseCgthr(
        handle, nnz, csrVal, (hipComplex*)pBuffer, info->P, HIPSPARSE_INDEX_BASE_ZERO));

    // Get stream
    hipStream_t stream;
    RETURN_IF_HIPSPARSE_ERROR(hipsparseGetStream(handle, &stream));

    // Copy sorted values back to csrVal
    RETURN_IF_HIP_ERROR(
        hipMemcpyAsync(csrVal, pBuffer, sizeof(hipComplex) * nnz, hipMemcpyDeviceToDevice, stream));

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseZcsru2csr(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       n,
                                     int                       nnz,
                                     const hipsparseMatDescr_t descrA,
                                     hipDoubleComplex*         csrVal,
                                     const int*                csrRowPtr,
                                     int*                      csrColInd,
                                     csru2csrInfo_t            info,
                                     void*                     pBuffer)
{
    // Test for bad args
    if(handle == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    // Invalid sizes
    if(m < 0 || n < 0 || nnz < 0)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    // Quick return
    if(m == 0 || n == 0 || nnz == 0)
    {
        // nnz must be 0
        if(nnz != 0)
        {
            return HIPSPARSE_STATUS_INVALID_VALUE;
        }

        return HIPSPARSE_STATUS_SUCCESS;
    }

    // Invalid pointers
    if(descrA == nullptr || csrVal == nullptr || csrRowPtr == nullptr || csrColInd == nullptr
       || info == nullptr || pBuffer == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    // De-allocate permutation array, if already allocated but sizes do not match
    if(info->P != nullptr && info->size != nnz)
    {
        RETURN_IF_HIP_ERROR(hipFree(info->P));
        info->size = 0;
    }

    // Allocate memory inside info structure to keep track of the permutation
    // if it has not yet been allocated with matching size
    if(info->P == nullptr)
    {
        // size must be 0
        assert(info->size == 0);

        RETURN_IF_HIP_ERROR(hipMalloc((void**)&info->P, sizeof(int) * nnz));

        info->size = nnz;
    }

    // Initialize permutation with identity
    RETURN_IF_HIPSPARSE_ERROR(hipsparseCreateIdentityPermutation(handle, nnz, info->P));

    // Sort CSR columns
    RETURN_IF_HIPSPARSE_ERROR(
        hipsparseXcsrsort(handle, m, n, nnz, descrA, csrRowPtr, csrColInd, info->P, pBuffer));

    // Sort CSR values
    RETURN_IF_HIPSPARSE_ERROR(hipsparseZgthr(
        handle, nnz, csrVal, (hipDoubleComplex*)pBuffer, info->P, HIPSPARSE_INDEX_BASE_ZERO));

    // Get stream
    hipStream_t stream;
    RETURN_IF_HIPSPARSE_ERROR(hipsparseGetStream(handle, &stream));

    // Copy sorted values back to csrVal
    RETURN_IF_HIP_ERROR(hipMemcpyAsync(
        csrVal, pBuffer, sizeof(hipDoubleComplex) * nnz, hipMemcpyDeviceToDevice, stream));

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseScsr2csru(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       n,
                                     int                       nnz,
                                     const hipsparseMatDescr_t descrA,
                                     float*                    csrVal,
                                     const int*                csrRowPtr,
                                     int*                      csrColInd,
                                     csru2csrInfo_t            info,
                                     void*                     pBuffer)
{
    // Test for bad args
    if(handle == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    // Invalid sizes
    if(m < 0 || n < 0 || nnz < 0)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    // Quick return
    if(m == 0 || n == 0 || nnz == 0)
    {
        // nnz must be 0
        if(nnz != 0)
        {
            return HIPSPARSE_STATUS_INVALID_VALUE;
        }

        return HIPSPARSE_STATUS_SUCCESS;
    }

    // Invalid pointers
    if(descrA == nullptr || csrVal == nullptr || csrRowPtr == nullptr || csrColInd == nullptr
       || info == nullptr || pBuffer == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    // Check for valid permutation array
    if(info->P == nullptr || info->size != nnz)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    // Get stream
    hipStream_t stream;
    RETURN_IF_HIPSPARSE_ERROR(hipsparseGetStream(handle, &stream));

    // Unsort CSR column indices based on the given permutation
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_isctr((rocsparse_handle)handle,
                                              nnz,
                                              csrColInd,
                                              info->P,
                                              (int*)pBuffer,
                                              rocsparse_index_base_zero));

    // Copy unsorted column indices back to csrColInd
    RETURN_IF_HIP_ERROR(
        hipMemcpyAsync(csrColInd, pBuffer, sizeof(int) * nnz, hipMemcpyDeviceToDevice, stream));

    // Unsort CSR values based on the given permutation
    RETURN_IF_HIPSPARSE_ERROR(
        hipsparseSsctr(handle, nnz, csrVal, info->P, (float*)pBuffer, HIPSPARSE_INDEX_BASE_ZERO));

    // Copy unsorted values back to csrVal
    RETURN_IF_HIP_ERROR(
        hipMemcpyAsync(csrVal, pBuffer, sizeof(float) * nnz, hipMemcpyDeviceToDevice, stream));

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseDcsr2csru(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       n,
                                     int                       nnz,
                                     const hipsparseMatDescr_t descrA,
                                     double*                   csrVal,
                                     const int*                csrRowPtr,
                                     int*                      csrColInd,
                                     csru2csrInfo_t            info,
                                     void*                     pBuffer)
{
    // Test for bad args
    if(handle == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    // Invalid sizes
    if(m < 0 || n < 0 || nnz < 0)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    // Quick return
    if(m == 0 || n == 0 || nnz == 0)
    {
        // nnz must be 0
        if(nnz != 0)
        {
            return HIPSPARSE_STATUS_INVALID_VALUE;
        }

        return HIPSPARSE_STATUS_SUCCESS;
    }

    // Invalid pointers
    if(descrA == nullptr || csrVal == nullptr || csrRowPtr == nullptr || csrColInd == nullptr
       || info == nullptr || pBuffer == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    // Check for valid permutation array
    if(info->P == nullptr || info->size != nnz)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    // Get stream
    hipStream_t stream;
    RETURN_IF_HIPSPARSE_ERROR(hipsparseGetStream(handle, &stream));

    // Unsort CSR column indices based on the given permutation
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_isctr((rocsparse_handle)handle,
                                              nnz,
                                              csrColInd,
                                              info->P,
                                              (int*)pBuffer,
                                              rocsparse_index_base_zero));

    // Copy unsorted column indices back to csrColInd
    RETURN_IF_HIP_ERROR(
        hipMemcpyAsync(csrColInd, pBuffer, sizeof(int) * nnz, hipMemcpyDeviceToDevice, stream));

    // Unsort CSR values based on the given permutation
    RETURN_IF_HIPSPARSE_ERROR(
        hipsparseDsctr(handle, nnz, csrVal, info->P, (double*)pBuffer, HIPSPARSE_INDEX_BASE_ZERO));

    // Copy unsorted values back to csrVal
    RETURN_IF_HIP_ERROR(
        hipMemcpyAsync(csrVal, pBuffer, sizeof(double) * nnz, hipMemcpyDeviceToDevice, stream));

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseCcsr2csru(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       n,
                                     int                       nnz,
                                     const hipsparseMatDescr_t descrA,
                                     hipComplex*               csrVal,
                                     const int*                csrRowPtr,
                                     int*                      csrColInd,
                                     csru2csrInfo_t            info,
                                     void*                     pBuffer)
{
    // Test for bad args
    if(handle == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    // Invalid sizes
    if(m < 0 || n < 0 || nnz < 0)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    // Quick return
    if(m == 0 || n == 0 || nnz == 0)
    {
        // nnz must be 0
        if(nnz != 0)
        {
            return HIPSPARSE_STATUS_INVALID_VALUE;
        }

        return HIPSPARSE_STATUS_SUCCESS;
    }

    // Invalid pointers
    if(descrA == nullptr || csrVal == nullptr || csrRowPtr == nullptr || csrColInd == nullptr
       || info == nullptr || pBuffer == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    // Check for valid permutation array
    if(info->P == nullptr || info->size != nnz)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    // Get stream
    hipStream_t stream;
    RETURN_IF_HIPSPARSE_ERROR(hipsparseGetStream(handle, &stream));

    // Unsort CSR column indices based on the given permutation
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_isctr((rocsparse_handle)handle,
                                              nnz,
                                              csrColInd,
                                              info->P,
                                              (int*)pBuffer,
                                              rocsparse_index_base_zero));

    // Copy unsorted column indices back to csrColInd
    RETURN_IF_HIP_ERROR(
        hipMemcpyAsync(csrColInd, pBuffer, sizeof(int) * nnz, hipMemcpyDeviceToDevice, stream));

    // Unsort CSR values based on the given permutation
    RETURN_IF_HIPSPARSE_ERROR(hipsparseCsctr(
        handle, nnz, csrVal, info->P, (hipComplex*)pBuffer, HIPSPARSE_INDEX_BASE_ZERO));

    // Copy unsorted values back to csrVal
    RETURN_IF_HIP_ERROR(
        hipMemcpyAsync(csrVal, pBuffer, sizeof(hipComplex) * nnz, hipMemcpyDeviceToDevice, stream));

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseZcsr2csru(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       n,
                                     int                       nnz,
                                     const hipsparseMatDescr_t descrA,
                                     hipDoubleComplex*         csrVal,
                                     const int*                csrRowPtr,
                                     int*                      csrColInd,
                                     csru2csrInfo_t            info,
                                     void*                     pBuffer)
{
    // Test for bad args
    if(handle == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    // Invalid sizes
    if(m < 0 || n < 0 || nnz < 0)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    // Quick return
    if(m == 0 || n == 0 || nnz == 0)
    {
        // nnz must be 0
        if(nnz != 0)
        {
            return HIPSPARSE_STATUS_INVALID_VALUE;
        }

        return HIPSPARSE_STATUS_SUCCESS;
    }

    // Invalid pointers
    if(descrA == nullptr || csrVal == nullptr || csrRowPtr == nullptr || csrColInd == nullptr
       || info == nullptr || pBuffer == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    // Check for valid permutation array
    if(info->P == nullptr || info->size != nnz)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    // Get stream
    hipStream_t stream;
    RETURN_IF_HIPSPARSE_ERROR(hipsparseGetStream(handle, &stream));

    // Unsort CSR column indices based on the given permutation
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_isctr((rocsparse_handle)handle,
                                              nnz,
                                              csrColInd,
                                              info->P,
                                              (int*)pBuffer,
                                              rocsparse_index_base_zero));

    // Copy unsorted column indices back to csrColInd
    RETURN_IF_HIP_ERROR(
        hipMemcpyAsync(csrColInd, pBuffer, sizeof(int) * nnz, hipMemcpyDeviceToDevice, stream));

    // Unsort CSR values based on the given permutation
    RETURN_IF_HIPSPARSE_ERROR(hipsparseZsctr(
        handle, nnz, csrVal, info->P, (hipDoubleComplex*)pBuffer, HIPSPARSE_INDEX_BASE_ZERO));

    // Copy unsorted values back to csrVal
    RETURN_IF_HIP_ERROR(hipMemcpyAsync(
        csrVal, pBuffer, sizeof(hipDoubleComplex) * nnz, hipMemcpyDeviceToDevice, stream));

    return HIPSPARSE_STATUS_SUCCESS;
}

/* Generic API */
hipsparseStatus_t hipsparseCreateSpVec(hipsparseSpVecDescr_t* spVecDescr,
                                       int64_t                size,
                                       int64_t                nnz,
                                       void*                  indices,
                                       void*                  values,
                                       hipsparseIndexType_t   idxType,
                                       hipsparseIndexBase_t   idxBase,
                                       hipDataType            valueType)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_create_spvec_descr((rocsparse_spvec_descr*)spVecDescr,
                                     size,
                                     nnz,
                                     indices,
                                     values,
                                     hipIndexTypeToHCCIndexType(idxType),
                                     hipBaseToHCCBase(idxBase),
                                     hipDataTypeToHCCDataType(valueType)));
}

hipsparseStatus_t hipsparseDestroySpVec(hipsparseSpVecDescr_t spVecDescr)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_destroy_spvec_descr((rocsparse_spvec_descr)spVecDescr));
}

hipsparseStatus_t hipsparseSpVecGet(const hipsparseSpVecDescr_t spVecDescr,
                                    int64_t*                    size,
                                    int64_t*                    nnz,
                                    void**                      indices,
                                    void**                      values,
                                    hipsparseIndexType_t*       idxType,
                                    hipsparseIndexBase_t*       idxBase,
                                    hipDataType*                valueType)
{
    rocsparse_indextype  hcc_index_type;
    rocsparse_index_base hcc_index_base;
    rocsparse_datatype   hcc_data_type;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse_spvec_get((const rocsparse_spvec_descr)spVecDescr,
                                                  size,
                                                  nnz,
                                                  indices,
                                                  values,
                                                  idxType != nullptr ? &hcc_index_type : nullptr,
                                                  idxBase != nullptr ? &hcc_index_base : nullptr,
                                                  valueType != nullptr ? &hcc_data_type : nullptr));

    *idxType   = HCCIndexTypeToHIPIndexType(hcc_index_type);
    *idxBase   = HCCBaseToHIPBase(hcc_index_base);
    *valueType = HCCDataTypeToHIPDataType(hcc_data_type);

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseSpVecGetIndexBase(const hipsparseSpVecDescr_t spVecDescr,
                                             hipsparseIndexBase_t*       idxBase)
{
    rocsparse_index_base hcc_index_base;
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_spvec_get_index_base(
        (const rocsparse_spvec_descr)spVecDescr, idxBase != nullptr ? &hcc_index_base : nullptr));

    *idxBase = HCCBaseToHIPBase(hcc_index_base);

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseSpVecGetValues(const hipsparseSpVecDescr_t spVecDescr, void** values)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_spvec_get_values((const rocsparse_spvec_descr)spVecDescr, values));
}

hipsparseStatus_t hipsparseSpVecSetValues(hipsparseSpVecDescr_t spVecDescr, void* values)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_spvec_set_values((rocsparse_spvec_descr)spVecDescr, values));
}

hipsparseStatus_t hipsparseCreateCoo(hipsparseSpMatDescr_t* spMatDescr,
                                     int64_t                rows,
                                     int64_t                cols,
                                     int64_t                nnz,
                                     void*                  cooRowInd,
                                     void*                  cooColInd,
                                     void*                  cooValues,
                                     hipsparseIndexType_t   cooIdxType,
                                     hipsparseIndexBase_t   idxBase,
                                     hipDataType            valueType)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_create_coo_descr((rocsparse_spmat_descr*)spMatDescr,
                                   rows,
                                   cols,
                                   nnz,
                                   cooRowInd,
                                   cooColInd,
                                   cooValues,
                                   hipIndexTypeToHCCIndexType(cooIdxType),
                                   hipBaseToHCCBase(idxBase),
                                   hipDataTypeToHCCDataType(valueType)));
}

hipsparseStatus_t hipsparseCreateCooAoS(hipsparseSpMatDescr_t* spMatDescr,
                                        int64_t                rows,
                                        int64_t                cols,
                                        int64_t                nnz,
                                        void*                  cooInd,
                                        void*                  cooValues,
                                        hipsparseIndexType_t   cooIdxType,
                                        hipsparseIndexBase_t   idxBase,
                                        hipDataType            valueType)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_create_coo_aos_descr((rocsparse_spmat_descr*)spMatDescr,
                                       rows,
                                       cols,
                                       nnz,
                                       cooInd,
                                       cooValues,
                                       hipIndexTypeToHCCIndexType(cooIdxType),
                                       hipBaseToHCCBase(idxBase),
                                       hipDataTypeToHCCDataType(valueType)));
}

hipsparseStatus_t hipsparseCreateCsr(hipsparseSpMatDescr_t* spMatDescr,
                                     int64_t                rows,
                                     int64_t                cols,
                                     int64_t                nnz,
                                     void*                  csrRowOffsets,
                                     void*                  csrColInd,
                                     void*                  csrValues,
                                     hipsparseIndexType_t   csrRowOffsetsType,
                                     hipsparseIndexType_t   csrColIndType,
                                     hipsparseIndexBase_t   idxBase,
                                     hipDataType            valueType)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_create_csr_descr((rocsparse_spmat_descr*)spMatDescr,
                                   rows,
                                   cols,
                                   nnz,
                                   csrRowOffsets,
                                   csrColInd,
                                   csrValues,
                                   hipIndexTypeToHCCIndexType(csrRowOffsetsType),
                                   hipIndexTypeToHCCIndexType(csrColIndType),
                                   hipBaseToHCCBase(idxBase),
                                   hipDataTypeToHCCDataType(valueType)));
}

hipsparseStatus_t hipsparseCreateCsc(hipsparseSpMatDescr_t* spMatDescr,
                                     int64_t                rows,
                                     int64_t                cols,
                                     int64_t                nnz,
                                     void*                  cscColOffsets,
                                     void*                  cscRowInd,
                                     void*                  cscValues,
                                     hipsparseIndexType_t   cscColOffsetsType,
                                     hipsparseIndexType_t   cscRowIndType,
                                     hipsparseIndexBase_t   idxBase,
                                     hipDataType            valueType)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_create_csc_descr((rocsparse_spmat_descr*)spMatDescr,
                                   rows,
                                   cols,
                                   nnz,
                                   cscColOffsets,
                                   cscRowInd,
                                   cscValues,
                                   hipIndexTypeToHCCIndexType(cscColOffsetsType),
                                   hipIndexTypeToHCCIndexType(cscRowIndType),
                                   hipBaseToHCCBase(idxBase),
                                   hipDataTypeToHCCDataType(valueType)));
}

hipsparseStatus_t hipsparseDestroySpMat(hipsparseSpMatDescr_t spMatDescr)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_destroy_spmat_descr((rocsparse_spmat_descr)spMatDescr));
}

hipsparseStatus_t hipsparseCooGet(const hipsparseSpMatDescr_t spMatDescr,
                                  int64_t*                    rows,
                                  int64_t*                    cols,
                                  int64_t*                    nnz,
                                  void**                      cooRowInd,
                                  void**                      cooColInd,
                                  void**                      cooValues,
                                  hipsparseIndexType_t*       idxType,
                                  hipsparseIndexBase_t*       idxBase,
                                  hipDataType*                valueType)
{
    rocsparse_indextype  hcc_index_type;
    rocsparse_index_base hcc_index_base;
    rocsparse_datatype   hcc_data_type;

    RETURN_IF_ROCSPARSE_ERROR(rocsparse_coo_get((const rocsparse_spmat_descr)spMatDescr,
                                                rows,
                                                cols,
                                                nnz,
                                                cooRowInd,
                                                cooColInd,
                                                cooValues,
                                                idxType != nullptr ? &hcc_index_type : nullptr,
                                                idxBase != nullptr ? &hcc_index_base : nullptr,
                                                valueType != nullptr ? &hcc_data_type : nullptr));

    *idxType   = HCCIndexTypeToHIPIndexType(hcc_index_type);
    *idxBase   = HCCBaseToHIPBase(hcc_index_base);
    *valueType = HCCDataTypeToHIPDataType(hcc_data_type);

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseCooAoSGet(const hipsparseSpMatDescr_t spMatDescr,
                                     int64_t*                    rows,
                                     int64_t*                    cols,
                                     int64_t*                    nnz,
                                     void**                      cooInd,
                                     void**                      cooValues,
                                     hipsparseIndexType_t*       idxType,
                                     hipsparseIndexBase_t*       idxBase,
                                     hipDataType*                valueType)
{
    rocsparse_indextype  hcc_index_type;
    rocsparse_index_base hcc_index_base;
    rocsparse_datatype   hcc_data_type;

    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse_coo_aos_get((const rocsparse_spmat_descr)spMatDescr,
                              rows,
                              cols,
                              nnz,
                              cooInd,
                              cooValues,
                              idxType != nullptr ? &hcc_index_type : nullptr,
                              idxBase != nullptr ? &hcc_index_base : nullptr,
                              valueType != nullptr ? &hcc_data_type : nullptr));

    *idxType   = HCCIndexTypeToHIPIndexType(hcc_index_type);
    *idxBase   = HCCBaseToHIPBase(hcc_index_base);
    *valueType = HCCDataTypeToHIPDataType(hcc_data_type);

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseCsrGet(const hipsparseSpMatDescr_t spMatDescr,
                                  int64_t*                    rows,
                                  int64_t*                    cols,
                                  int64_t*                    nnz,
                                  void**                      csrRowOffsets,
                                  void**                      csrColInd,
                                  void**                      csrValues,
                                  hipsparseIndexType_t*       csrRowOffsetsType,
                                  hipsparseIndexType_t*       csrColIndType,
                                  hipsparseIndexBase_t*       idxBase,
                                  hipDataType*                valueType)
{
    rocsparse_indextype  hcc_row_index_type;
    rocsparse_indextype  hcc_col_index_type;
    rocsparse_index_base hcc_index_base;
    rocsparse_datatype   hcc_data_type;

    RETURN_IF_ROCSPARSE_ERROR(
        rocsparse_csr_get((const rocsparse_spmat_descr)spMatDescr,
                          rows,
                          cols,
                          nnz,
                          csrRowOffsets,
                          csrColInd,
                          csrValues,
                          csrRowOffsetsType != nullptr ? &hcc_row_index_type : nullptr,
                          csrColIndType != nullptr ? &hcc_col_index_type : nullptr,
                          idxBase != nullptr ? &hcc_index_base : nullptr,
                          valueType != nullptr ? &hcc_data_type : nullptr));

    *csrRowOffsetsType = HCCIndexTypeToHIPIndexType(hcc_row_index_type);
    *csrColIndType     = HCCIndexTypeToHIPIndexType(hcc_col_index_type);
    *idxBase           = HCCBaseToHIPBase(hcc_index_base);
    *valueType         = HCCDataTypeToHIPDataType(hcc_data_type);

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseCsrSetPointers(hipsparseSpMatDescr_t spMatDescr,
                                          void*                 csrRowOffsets,
                                          void*                 csrColInd,
                                          void*                 csrValues)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_csr_set_pointers(
        (rocsparse_spmat_descr)spMatDescr, csrRowOffsets, csrColInd, csrValues));
}

hipsparseStatus_t hipsparseCscSetPointers(hipsparseSpMatDescr_t spMatDescr,
                                          void*                 cscColOffsets,
                                          void*                 cscRowInd,
                                          void*                 cscValues)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_csc_set_pointers(
        (rocsparse_spmat_descr)spMatDescr, cscColOffsets, cscRowInd, cscValues));
}

hipsparseStatus_t hipsparseCooSetPointers(hipsparseSpMatDescr_t spMatDescr,
                                          void*                 cooRowInd,
                                          void*                 cooColInd,
                                          void*                 cooValues)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_coo_set_pointers(
        (rocsparse_spmat_descr)spMatDescr, cooRowInd, cooColInd, cooValues));
}

hipsparseStatus_t hipsparseSpMatGetSize(hipsparseSpMatDescr_t spMatDescr,
                                        int64_t*              rows,
                                        int64_t*              cols,
                                        int64_t*              nnz)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_spmat_get_size((rocsparse_spmat_descr)spMatDescr, rows, cols, nnz));
}

hipsparseStatus_t hipsparseSpMatGetFormat(const hipsparseSpMatDescr_t spMatDescr,
                                          hipsparseFormat_t*          format)
{
    rocsparse_format hcc_format;
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_spmat_get_format(
        (const rocsparse_spmat_descr)spMatDescr, format != nullptr ? &hcc_format : nullptr));

    *format = HCCFormatToHIPFormat(hcc_format);

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseSpMatGetIndexBase(const hipsparseSpMatDescr_t spMatDescr,
                                             hipsparseIndexBase_t*       idxBase)
{
    rocsparse_index_base hcc_index_base;
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_spmat_get_index_base(
        (const rocsparse_spmat_descr)spMatDescr, idxBase != nullptr ? &hcc_index_base : nullptr));

    *idxBase = HCCBaseToHIPBase(hcc_index_base);

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseSpMatGetValues(hipsparseSpMatDescr_t spMatDescr, void** values)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_spmat_get_values((rocsparse_spmat_descr)spMatDescr, values));
}

hipsparseStatus_t hipsparseSpMatSetValues(hipsparseSpMatDescr_t spMatDescr, void* values)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_spmat_set_values((rocsparse_spmat_descr)spMatDescr, values));
}

hipsparseStatus_t hipsparseCreateDnVec(hipsparseDnVecDescr_t* dnVecDescr,
                                       int64_t                size,
                                       void*                  values,
                                       hipDataType            valueType)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_create_dnvec_descr(
        (rocsparse_dnvec_descr*)dnVecDescr, size, values, hipDataTypeToHCCDataType(valueType)));
}

hipsparseStatus_t hipsparseDestroyDnVec(hipsparseDnVecDescr_t dnVecDescr)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_destroy_dnvec_descr((rocsparse_dnvec_descr)dnVecDescr));
}

hipsparseStatus_t hipsparseDnVecGet(const hipsparseDnVecDescr_t dnVecDescr,
                                    int64_t*                    size,
                                    void**                      values,
                                    hipDataType*                valueType)
{
    rocsparse_datatype hcc_data_type;
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_dnvec_get((const rocsparse_dnvec_descr)dnVecDescr,
                                                  size,
                                                  values,
                                                  valueType != nullptr ? &hcc_data_type : nullptr));

    *valueType = HCCDataTypeToHIPDataType(hcc_data_type);

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseDnVecGetValues(const hipsparseDnVecDescr_t dnVecDescr, void** values)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_dnvec_get_values((const rocsparse_dnvec_descr)dnVecDescr, values));
}

hipsparseStatus_t hipsparseDnVecSetValues(hipsparseDnVecDescr_t dnVecDescr, void* values)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_dnvec_set_values((rocsparse_dnvec_descr)dnVecDescr, values));
}

hipsparseStatus_t hipsparseCreateDnMat(hipsparseDnMatDescr_t* dnMatDescr,
                                       int64_t                rows,
                                       int64_t                cols,
                                       int64_t                ld,
                                       void*                  values,
                                       hipDataType            valueType,
                                       hipsparseOrder_t       order)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_create_dnmat_descr((rocsparse_dnmat_descr*)dnMatDescr,
                                     rows,
                                     cols,
                                     ld,
                                     values,
                                     hipDataTypeToHCCDataType(valueType),
                                     hipOrderToHCCOrder(order)));
}

hipsparseStatus_t hipsparseDestroyDnMat(hipsparseDnMatDescr_t dnMatDescr)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_destroy_dnmat_descr((rocsparse_dnmat_descr)dnMatDescr));
}

hipsparseStatus_t hipsparseDnMatGet(const hipsparseDnMatDescr_t dnMatDescr,
                                    int64_t*                    rows,
                                    int64_t*                    cols,
                                    int64_t*                    ld,
                                    void**                      values,
                                    hipDataType*                valueType,
                                    hipsparseOrder_t*           order)
{
    rocsparse_datatype hcc_data_type;
    rocsparse_order    hcc_order;
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_dnmat_get((const rocsparse_dnmat_descr)dnMatDescr,
                                                  rows,
                                                  cols,
                                                  ld,
                                                  values,
                                                  valueType != nullptr ? &hcc_data_type : nullptr,
                                                  order != nullptr ? &hcc_order : nullptr));

    *valueType = HCCDataTypeToHIPDataType(hcc_data_type);
    *order     = HCCOrderToHIPOrder(hcc_order);

    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseDnMatGetValues(const hipsparseDnMatDescr_t dnMatDescr, void** values)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_dnmat_get_values((const rocsparse_dnmat_descr)dnMatDescr, values));
}

hipsparseStatus_t hipsparseDnMatSetValues(hipsparseDnMatDescr_t dnMatDescr, void* values)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_dnmat_set_values((rocsparse_dnmat_descr)dnMatDescr, values));
}

hipsparseStatus_t hipsparseAxpby(hipsparseHandle_t     handle,
                                 const void*           alpha,
                                 hipsparseSpVecDescr_t vecX,
                                 const void*           beta,
                                 hipsparseDnVecDescr_t vecY)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_axpby((rocsparse_handle)handle,
                                                      alpha,
                                                      (rocsparse_spvec_descr)vecX,
                                                      beta,
                                                      (rocsparse_dnvec_descr)vecY));
}

hipsparseStatus_t hipsparseGather(hipsparseHandle_t     handle,
                                  hipsparseDnVecDescr_t vecY,
                                  hipsparseSpVecDescr_t vecX)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_gather(
        (rocsparse_handle)handle, (rocsparse_dnvec_descr)vecY, (rocsparse_spvec_descr)vecX));
}

hipsparseStatus_t hipsparseScatter(hipsparseHandle_t     handle,
                                   hipsparseSpVecDescr_t vecX,
                                   hipsparseDnVecDescr_t vecY)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_scatter(
        (rocsparse_handle)handle, (rocsparse_spvec_descr)vecX, (rocsparse_dnvec_descr)vecY));
}

hipsparseStatus_t hipsparseRot(hipsparseHandle_t     handle,
                               const void*           c_coeff,
                               const void*           s_coeff,
                               hipsparseSpVecDescr_t vecX,
                               hipsparseDnVecDescr_t vecY)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_rot((rocsparse_handle)handle,
                                                    c_coeff,
                                                    s_coeff,
                                                    (rocsparse_spvec_descr)vecX,
                                                    (rocsparse_dnvec_descr)vecY));
}

hipsparseStatus_t hipsparseSparseToDense_bufferSize(hipsparseHandle_t           handle,
                                                    hipsparseSpMatDescr_t       matA,
                                                    hipsparseDnMatDescr_t       matB,
                                                    hipsparseSparseToDenseAlg_t alg,
                                                    size_t*                     bufferSize)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_sparse_to_dense((rocsparse_handle)handle,
                                                                (rocsparse_spmat_descr)matA,
                                                                (rocsparse_dnmat_descr)matB,
                                                                hipSpToDnAlgToHCCSpToDnAlg(alg),
                                                                bufferSize,
                                                                nullptr));
}

hipsparseStatus_t hipsparseSparseToDense(hipsparseHandle_t           handle,
                                         hipsparseSpMatDescr_t       matA,
                                         hipsparseDnMatDescr_t       matB,
                                         hipsparseSparseToDenseAlg_t alg,
                                         void*                       externalBuffer)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_sparse_to_dense((rocsparse_handle)handle,
                                                                (rocsparse_spmat_descr)matA,
                                                                (rocsparse_dnmat_descr)matB,
                                                                hipSpToDnAlgToHCCSpToDnAlg(alg),
                                                                nullptr,
                                                                externalBuffer));
}

hipsparseStatus_t hipsparseDenseToSparse_bufferSize(hipsparseHandle_t           handle,
                                                    hipsparseDnMatDescr_t       matA,
                                                    hipsparseSpMatDescr_t       matB,
                                                    hipsparseDenseToSparseAlg_t alg,
                                                    size_t*                     bufferSize)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_dense_to_sparse((rocsparse_handle)handle,
                                                                (rocsparse_dnmat_descr)matA,
                                                                (rocsparse_spmat_descr)matB,
                                                                hipDnToSpAlgToHCCDnToSpAlg(alg),
                                                                bufferSize,
                                                                nullptr));
}

hipsparseStatus_t hipsparseDenseToSparse_analysis(hipsparseHandle_t           handle,
                                                  hipsparseDnMatDescr_t       matA,
                                                  hipsparseSpMatDescr_t       matB,
                                                  hipsparseDenseToSparseAlg_t alg,
                                                  void*                       externalBuffer)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_dense_to_sparse((rocsparse_handle)handle,
                                                                (rocsparse_dnmat_descr)matA,
                                                                (rocsparse_spmat_descr)matB,
                                                                hipDnToSpAlgToHCCDnToSpAlg(alg),
                                                                nullptr,
                                                                externalBuffer));
}

hipsparseStatus_t hipsparseDenseToSparse_convert(hipsparseHandle_t           handle,
                                                 hipsparseDnMatDescr_t       matA,
                                                 hipsparseSpMatDescr_t       matB,
                                                 hipsparseDenseToSparseAlg_t alg,
                                                 void*                       externalBuffer)
{
    size_t bufferSize = 4;
    return rocSPARSEStatusToHIPStatus(
        rocsparse_dense_to_sparse((rocsparse_handle)handle,
                                  (rocsparse_dnmat_descr)matA,
                                  (rocsparse_spmat_descr)matB,
                                  hipDnToSpAlgToHCCDnToSpAlg(alg),
                                  externalBuffer != nullptr ? &bufferSize : nullptr,
                                  externalBuffer));
}

hipsparseStatus_t hipsparseSpVV_bufferSize(hipsparseHandle_t     handle,
                                           hipsparseOperation_t  opX,
                                           hipsparseSpVecDescr_t vecX,
                                           hipsparseDnVecDescr_t vecY,
                                           void*                 result,
                                           hipDataType           computeType,
                                           size_t*               bufferSize)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_spvv((rocsparse_handle)handle,
                                                     hipOperationToHCCOperation(opX),
                                                     (rocsparse_spvec_descr)vecX,
                                                     (rocsparse_dnvec_descr)vecY,
                                                     result,
                                                     hipDataTypeToHCCDataType(computeType),
                                                     bufferSize,
                                                     nullptr));
}

hipsparseStatus_t hipsparseSpVV(hipsparseHandle_t     handle,
                                hipsparseOperation_t  opX,
                                hipsparseSpVecDescr_t vecX,
                                hipsparseDnVecDescr_t vecY,
                                void*                 result,
                                hipDataType           computeType,
                                void*                 externalBuffer)
{
    size_t bufferSize;

    // Check for buffer == nullptr as this is not done in rocsparse
    if(externalBuffer == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    return rocSPARSEStatusToHIPStatus(rocsparse_spvv((rocsparse_handle)handle,
                                                     hipOperationToHCCOperation(opX),
                                                     (rocsparse_spvec_descr)vecX,
                                                     (rocsparse_dnvec_descr)vecY,
                                                     result,
                                                     hipDataTypeToHCCDataType(computeType),
                                                     &bufferSize,
                                                     externalBuffer));
}

hipsparseStatus_t hipsparseSpMV_bufferSize(hipsparseHandle_t           handle,
                                           hipsparseOperation_t        opA,
                                           const void*                 alpha,
                                           const hipsparseSpMatDescr_t matA,
                                           const hipsparseDnVecDescr_t vecX,
                                           const void*                 beta,
                                           const hipsparseDnVecDescr_t vecY,
                                           hipDataType                 computeType,
                                           hipsparseSpMVAlg_t          alg,
                                           size_t*                     bufferSize)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_spmv((rocsparse_handle)handle,
                                                     hipOperationToHCCOperation(opA),
                                                     alpha,
                                                     (const rocsparse_spmat_descr)matA,
                                                     (const rocsparse_dnvec_descr)vecX,
                                                     beta,
                                                     (const rocsparse_dnvec_descr)vecY,
                                                     hipDataTypeToHCCDataType(computeType),
                                                     hipSpMVAlgToHCCSpMVAlg(alg),
                                                     bufferSize,
                                                     nullptr));
}

hipsparseStatus_t hipsparseSpMV(hipsparseHandle_t           handle,
                                hipsparseOperation_t        opA,
                                const void*                 alpha,
                                const hipsparseSpMatDescr_t matA,
                                const hipsparseDnVecDescr_t vecX,
                                const void*                 beta,
                                const hipsparseDnVecDescr_t vecY,
                                hipDataType                 computeType,
                                hipsparseSpMVAlg_t          alg,
                                void*                       externalBuffer)
{
    size_t bufferSize;
    return rocSPARSEStatusToHIPStatus(rocsparse_spmv((rocsparse_handle)handle,
                                                     hipOperationToHCCOperation(opA),
                                                     alpha,
                                                     (const rocsparse_spmat_descr)matA,
                                                     (const rocsparse_dnvec_descr)vecX,
                                                     beta,
                                                     (const rocsparse_dnvec_descr)vecY,
                                                     hipDataTypeToHCCDataType(computeType),
                                                     hipSpMVAlgToHCCSpMVAlg(alg),
                                                     &bufferSize,
                                                     externalBuffer));
}

hipsparseStatus_t hipsparseSpMM_bufferSize(hipsparseHandle_t           handle,
                                           hipsparseOperation_t        opA,
                                           hipsparseOperation_t        opB,
                                           const void*                 alpha,
                                           const hipsparseSpMatDescr_t matA,
                                           const hipsparseDnMatDescr_t matB,
                                           const void*                 beta,
                                           const hipsparseDnMatDescr_t matC,
                                           hipDataType                 computeType,
                                           hipsparseSpMMAlg_t          alg,
                                           size_t*                     bufferSize)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_spmm_ex((rocsparse_handle)handle,
                                                        hipOperationToHCCOperation(opA),
                                                        hipOperationToHCCOperation(opB),
                                                        alpha,
                                                        (const rocsparse_spmat_descr)matA,
                                                        (const rocsparse_dnmat_descr)matB,
                                                        beta,
                                                        (const rocsparse_dnmat_descr)matC,
                                                        hipDataTypeToHCCDataType(computeType),
                                                        hipSpMMAlgToHCCSpMMAlg(alg),
                                                        rocsparse_spmm_stage_buffer_size,
                                                        bufferSize,
                                                        nullptr));
}

hipsparseStatus_t hipsparseSpMM_preprocess(hipsparseHandle_t           handle,
                                           hipsparseOperation_t        opA,
                                           hipsparseOperation_t        opB,
                                           const void*                 alpha,
                                           const hipsparseSpMatDescr_t matA,
                                           const hipsparseDnMatDescr_t matB,
                                           const void*                 beta,
                                           const hipsparseDnMatDescr_t matC,
                                           hipDataType                 computeType,
                                           hipsparseSpMMAlg_t          alg,
                                           void*                       externalBuffer)
{
    size_t bufferSize;
    return rocSPARSEStatusToHIPStatus(rocsparse_spmm_ex((rocsparse_handle)handle,
                                                        hipOperationToHCCOperation(opA),
                                                        hipOperationToHCCOperation(opB),
                                                        alpha,
                                                        (const rocsparse_spmat_descr)matA,
                                                        (const rocsparse_dnmat_descr)matB,
                                                        beta,
                                                        (const rocsparse_dnmat_descr)matC,
                                                        hipDataTypeToHCCDataType(computeType),
                                                        hipSpMMAlgToHCCSpMMAlg(alg),
                                                        rocsparse_spmm_stage_preprocess,
                                                        &bufferSize,
                                                        externalBuffer));
}

hipsparseStatus_t hipsparseSpMM(hipsparseHandle_t           handle,
                                hipsparseOperation_t        opA,
                                hipsparseOperation_t        opB,
                                const void*                 alpha,
                                const hipsparseSpMatDescr_t matA,
                                const hipsparseDnMatDescr_t matB,
                                const void*                 beta,
                                const hipsparseDnMatDescr_t matC,
                                hipDataType                 computeType,
                                hipsparseSpMMAlg_t          alg,
                                void*                       externalBuffer)
{
    size_t bufferSize;
    return rocSPARSEStatusToHIPStatus(rocsparse_spmm_ex((rocsparse_handle)handle,
                                                        hipOperationToHCCOperation(opA),
                                                        hipOperationToHCCOperation(opB),
                                                        alpha,
                                                        (const rocsparse_spmat_descr)matA,
                                                        (const rocsparse_dnmat_descr)matB,
                                                        beta,
                                                        (const rocsparse_dnmat_descr)matC,
                                                        hipDataTypeToHCCDataType(computeType),
                                                        hipSpMMAlgToHCCSpMMAlg(alg),
                                                        rocsparse_spmm_stage_compute,
                                                        &bufferSize,
                                                        externalBuffer));
}

hipsparseStatus_t hipsparseSpGEMM_createDescr(hipsparseSpGEMMDescr_t* descr)
{
    // Do nothing
    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseSpGEMM_destroyDescr(hipsparseSpGEMMDescr_t descr)
{
    // Do nothing
    return HIPSPARSE_STATUS_SUCCESS;
}

hipsparseStatus_t hipsparseSpGEMM_workEstimation(hipsparseHandle_t      handle,
                                                 hipsparseOperation_t   opA,
                                                 hipsparseOperation_t   opB,
                                                 const void*            alpha,
                                                 hipsparseSpMatDescr_t  matA,
                                                 hipsparseSpMatDescr_t  matB,
                                                 const void*            beta,
                                                 hipsparseSpMatDescr_t  matC,
                                                 hipDataType            computeType,
                                                 hipsparseSpGEMMAlg_t   alg,
                                                 hipsparseSpGEMMDescr_t spgemmDescr,
                                                 size_t*                bufferSize1,
                                                 void*                  externalBuffer1)
{
    // Match cusparse error handling
    if(handle == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    if(alpha == nullptr || beta == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    if(matA == nullptr || matB == nullptr || matC == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    if(bufferSize1 == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    // spgemmDescr can be nullptr

    // Do nothing
    *bufferSize1 = 4;

    return HIPSPARSE_STATUS_SUCCESS;
}

static const void*
    spgemm_get_ptr(hipsparsePointerMode_t mode, hipDataType computeType, const void* ptr)
{
    const void* cast_ptr = nullptr;

    if(mode == HIPSPARSE_POINTER_MODE_HOST)
    {
        if(computeType == HIP_R_32F)
            cast_ptr = (*(const float*)ptr != 0.0f) ? ptr : nullptr;
        if(computeType == HIP_R_64F)
            cast_ptr = (*(const double*)ptr != 0.0) ? ptr : nullptr;
        if(computeType == HIP_C_32F)
            cast_ptr = (*(const hipComplex*)ptr != make_hipComplex(0.0f, 0.0f)) ? ptr : nullptr;
        if(computeType == HIP_C_64F)
            cast_ptr = (*(const hipDoubleComplex*)ptr != make_hipDoubleComplex(0.0, 0.0)) ? ptr
                                                                                          : nullptr;
    }
    else
    {
        if(computeType == HIP_R_32F)
        {
            float host;
            hipMemcpy(&host, ptr, sizeof(float), hipMemcpyDeviceToHost);
            cast_ptr = (host != 0.0f) ? ptr : nullptr;
        }
        if(computeType == HIP_R_64F)
        {
            double host;
            hipMemcpy(&host, ptr, sizeof(double), hipMemcpyDeviceToHost);
            cast_ptr = (host != 0.0) ? ptr : nullptr;
        }
        if(computeType == HIP_C_32F)
        {
            hipComplex host;
            hipMemcpy(&host, ptr, sizeof(hipComplex), hipMemcpyDeviceToHost);
            cast_ptr = (host != make_hipComplex(0.0f, 0.0f)) ? ptr : nullptr;
        }
        if(computeType == HIP_C_64F)
        {
            hipDoubleComplex host;
            hipMemcpy(&host, ptr, sizeof(hipDoubleComplex), hipMemcpyDeviceToHost);
            cast_ptr = (host != make_hipDoubleComplex(0.0, 0.0)) ? ptr : nullptr;
        }
    }

    return cast_ptr;
}

hipsparseStatus_t hipsparseSpGEMM_compute(hipsparseHandle_t      handle,
                                          hipsparseOperation_t   opA,
                                          hipsparseOperation_t   opB,
                                          const void*            alpha,
                                          hipsparseSpMatDescr_t  matA,
                                          hipsparseSpMatDescr_t  matB,
                                          const void*            beta,
                                          hipsparseSpMatDescr_t  matC,
                                          hipDataType            computeType,
                                          hipsparseSpGEMMAlg_t   alg,
                                          hipsparseSpGEMMDescr_t spgemmDescr,
                                          size_t*                bufferSize2,
                                          void*                  externalBuffer2)
{
    if(handle == nullptr || bufferSize2 == nullptr || alpha == nullptr || beta == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    hipsparsePointerMode_t mode;
    RETURN_IF_HIPSPARSE_ERROR(hipsparseGetPointerMode(handle, &mode));

    const void* alpha_ptr = spgemm_get_ptr(mode, computeType, alpha);
    const void* beta_ptr  = spgemm_get_ptr(mode, computeType, beta);

    return rocSPARSEStatusToHIPStatus(rocsparse_spgemm((rocsparse_handle)handle,
                                                       hipOperationToHCCOperation(opA),
                                                       hipOperationToHCCOperation(opB),
                                                       alpha_ptr,
                                                       (rocsparse_spmat_descr)matA,
                                                       (rocsparse_spmat_descr)matB,
                                                       beta_ptr,
                                                       (rocsparse_spmat_descr)matC,
                                                       (rocsparse_spmat_descr)matC,
                                                       hipDataTypeToHCCDataType(computeType),
                                                       hipSpGEMMAlgToHCCSpGEMMAlg(alg),
                                                       rocsparse_spgemm_stage_auto,
                                                       bufferSize2,
                                                       externalBuffer2));
}

hipsparseStatus_t hipsparseSpGEMM_copy(hipsparseHandle_t      handle,
                                       hipsparseOperation_t   opA,
                                       hipsparseOperation_t   opB,
                                       const void*            alpha,
                                       hipsparseSpMatDescr_t  matA,
                                       hipsparseSpMatDescr_t  matB,
                                       const void*            beta,
                                       hipsparseSpMatDescr_t  matC,
                                       hipDataType            computeType,
                                       hipsparseSpGEMMAlg_t   alg,
                                       hipsparseSpGEMMDescr_t spgemmDescr)
{
    if(handle == nullptr || alpha == nullptr || beta == nullptr)
    {
        return HIPSPARSE_STATUS_INVALID_VALUE;
    }

    hipsparsePointerMode_t mode;
    RETURN_IF_HIPSPARSE_ERROR(hipsparseGetPointerMode(handle, &mode));

    const void* alpha_ptr = spgemm_get_ptr(mode, computeType, alpha);
    const void* beta_ptr  = spgemm_get_ptr(mode, computeType, beta);

    // cuSPARSE API does not carry over the temporary storage buffer, therefore
    // we need to allocate additional memory. This will lead to lower performance
    // and thus we highly recommand to use rocSPARSE API instead!!!

    // Query for required buffer size
    size_t bufferSize;
    RETURN_IF_ROCSPARSE_ERROR(rocsparse_spgemm((rocsparse_handle)handle,
                                               hipOperationToHCCOperation(opA),
                                               hipOperationToHCCOperation(opB),
                                               alpha_ptr,
                                               (rocsparse_spmat_descr)matA,
                                               (rocsparse_spmat_descr)matB,
                                               beta_ptr,
                                               (rocsparse_spmat_descr)matC,
                                               (rocsparse_spmat_descr)matC,
                                               hipDataTypeToHCCDataType(computeType),
                                               hipSpGEMMAlgToHCCSpGEMMAlg(alg),
                                               rocsparse_spgemm_stage_buffer_size,
                                               &bufferSize,
                                               nullptr));

    void* buffer;
    RETURN_IF_HIP_ERROR(hipMalloc(&buffer, bufferSize));

    hipsparseStatus_t status
        = rocSPARSEStatusToHIPStatus(rocsparse_spgemm((rocsparse_handle)handle,
                                                      hipOperationToHCCOperation(opA),
                                                      hipOperationToHCCOperation(opB),
                                                      alpha_ptr,
                                                      (rocsparse_spmat_descr)matA,
                                                      (rocsparse_spmat_descr)matB,
                                                      beta_ptr,
                                                      (rocsparse_spmat_descr)matC,
                                                      (rocsparse_spmat_descr)matC,
                                                      hipDataTypeToHCCDataType(computeType),
                                                      hipSpGEMMAlgToHCCSpGEMMAlg(alg),
                                                      rocsparse_spgemm_stage_compute,
                                                      &bufferSize,
                                                      buffer));

    RETURN_IF_HIP_ERROR(hipFree(buffer));

    return status;
}

hipsparseStatus_t hipsparseSDDMM(hipsparseHandle_t           handle,
                                 hipsparseOperation_t        opA,
                                 hipsparseOperation_t        opB,
                                 const void*                 alpha,
                                 const hipsparseDnMatDescr_t matA,
                                 const hipsparseDnMatDescr_t matB,
                                 const void*                 beta,
                                 hipsparseSpMatDescr_t       matC,
                                 hipDataType                 computeType,
                                 hipsparseSDDMMAlg_t         alg,
                                 void*                       tempBuffer)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_sddmm((rocsparse_handle)handle,
                                                      hipOperationToHCCOperation(opA),
                                                      hipOperationToHCCOperation(opB),
                                                      alpha,
                                                      (const rocsparse_dnmat_descr)matA,
                                                      (const rocsparse_dnmat_descr)matB,
                                                      beta,
                                                      (const rocsparse_spmat_descr)matC,
                                                      hipDataTypeToHCCDataType(computeType),
                                                      hipSDDMMAlgToHCCSDDMMAlg(alg),
                                                      tempBuffer));
}

hipsparseStatus_t hipsparseSDDMM_bufferSize(hipsparseHandle_t           handle,
                                            hipsparseOperation_t        opA,
                                            hipsparseOperation_t        opB,
                                            const void*                 alpha,
                                            const hipsparseDnMatDescr_t matA,
                                            const hipsparseDnMatDescr_t matB,
                                            const void*                 beta,
                                            hipsparseSpMatDescr_t       matC,
                                            hipDataType                 computeType,
                                            hipsparseSDDMMAlg_t         alg,
                                            size_t*                     bufferSize)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_sddmm_buffer_size((rocsparse_handle)handle,
                                    hipOperationToHCCOperation(opA),
                                    hipOperationToHCCOperation(opB),
                                    alpha,
                                    (const rocsparse_dnmat_descr)matA,
                                    (const rocsparse_dnmat_descr)matB,
                                    beta,
                                    (const rocsparse_spmat_descr)matC,
                                    hipDataTypeToHCCDataType(computeType),
                                    hipSDDMMAlgToHCCSDDMMAlg(alg),
                                    bufferSize));
}

hipsparseStatus_t hipsparseSDDMM_preprocess(hipsparseHandle_t           handle,
                                            hipsparseOperation_t        opA,
                                            hipsparseOperation_t        opB,
                                            const void*                 alpha,
                                            const hipsparseDnMatDescr_t matA,
                                            const hipsparseDnMatDescr_t matB,
                                            const void*                 beta,
                                            hipsparseSpMatDescr_t       matC,
                                            hipDataType                 computeType,
                                            hipsparseSDDMMAlg_t         alg,
                                            void*                       tempBuffer)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_sddmm_preprocess((rocsparse_handle)handle,
                                   hipOperationToHCCOperation(opA),
                                   hipOperationToHCCOperation(opB),
                                   alpha,
                                   (const rocsparse_dnmat_descr)matA,
                                   (const rocsparse_dnmat_descr)matB,
                                   beta,
                                   (const rocsparse_spmat_descr)matC,
                                   hipDataTypeToHCCDataType(computeType),
                                   hipSDDMMAlgToHCCSDDMMAlg(alg),
                                   tempBuffer));
}

hipsparseStatus_t hipsparseSgtsv2StridedBatch_bufferSizeExt(hipsparseHandle_t handle,
                                                            int               m,
                                                            const float*      dl,
                                                            const float*      d,
                                                            const float*      du,
                                                            const float*      x,
                                                            int               batchCount,
                                                            int               batchStride,
                                                            size_t*           pBufferSizeInBytes)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_sgtsv_no_pivot_strided_batch_buffer_size(
        (rocsparse_handle)handle, m, dl, d, du, x, batchCount, batchStride, pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseDgtsv2StridedBatch_bufferSizeExt(hipsparseHandle_t handle,
                                                            int               m,
                                                            const double*     dl,
                                                            const double*     d,
                                                            const double*     du,
                                                            const double*     x,
                                                            int               batchCount,
                                                            int               batchStride,
                                                            size_t*           pBufferSizeInBytes)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_dgtsv_no_pivot_strided_batch_buffer_size(
        (rocsparse_handle)handle, m, dl, d, du, x, batchCount, batchStride, pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseCgtsv2StridedBatch_bufferSizeExt(hipsparseHandle_t handle,
                                                            int               m,
                                                            const hipComplex* dl,
                                                            const hipComplex* d,
                                                            const hipComplex* du,
                                                            const hipComplex* x,
                                                            int               batchCount,
                                                            int               batchStride,
                                                            size_t*           pBufferSizeInBytes)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_cgtsv_no_pivot_strided_batch_buffer_size((rocsparse_handle)handle,
                                                           m,
                                                           (const rocsparse_float_complex*)dl,
                                                           (const rocsparse_float_complex*)d,
                                                           (const rocsparse_float_complex*)du,
                                                           (const rocsparse_float_complex*)x,
                                                           batchCount,
                                                           batchStride,
                                                           pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseZgtsv2StridedBatch_bufferSizeExt(hipsparseHandle_t       handle,
                                                            int                     m,
                                                            const hipDoubleComplex* dl,
                                                            const hipDoubleComplex* d,
                                                            const hipDoubleComplex* du,
                                                            const hipDoubleComplex* x,
                                                            int                     batchCount,
                                                            int                     batchStride,
                                                            size_t* pBufferSizeInBytes)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_zgtsv_no_pivot_strided_batch_buffer_size((rocsparse_handle)handle,
                                                           m,
                                                           (const rocsparse_double_complex*)dl,
                                                           (const rocsparse_double_complex*)d,
                                                           (const rocsparse_double_complex*)du,
                                                           (const rocsparse_double_complex*)x,
                                                           batchCount,
                                                           batchStride,
                                                           pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseSgtsv2StridedBatch(hipsparseHandle_t handle,
                                              int               m,
                                              const float*      dl,
                                              const float*      d,
                                              const float*      du,
                                              float*            x,
                                              int               batchCount,
                                              int               batchStride,
                                              void*             pBuffer)

{
    return rocSPARSEStatusToHIPStatus(rocsparse_sgtsv_no_pivot_strided_batch(
        (rocsparse_handle)handle, m, dl, d, du, x, batchCount, batchStride, pBuffer));
}

hipsparseStatus_t hipsparseDgtsv2StridedBatch(hipsparseHandle_t handle,
                                              int               m,
                                              const double*     dl,
                                              const double*     d,
                                              const double*     du,
                                              double*           x,
                                              int               batchCount,
                                              int               batchStride,
                                              void*             pBuffer)

{
    return rocSPARSEStatusToHIPStatus(rocsparse_dgtsv_no_pivot_strided_batch(
        (rocsparse_handle)handle, m, dl, d, du, x, batchCount, batchStride, pBuffer));
}

hipsparseStatus_t hipsparseCgtsv2StridedBatch(hipsparseHandle_t handle,
                                              int               m,
                                              const hipComplex* dl,
                                              const hipComplex* d,
                                              const hipComplex* du,
                                              hipComplex*       x,
                                              int               batchCount,
                                              int               batchStride,
                                              void*             pBuffer)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_cgtsv_no_pivot_strided_batch((rocsparse_handle)handle,
                                               m,
                                               (const rocsparse_float_complex*)dl,
                                               (const rocsparse_float_complex*)d,
                                               (const rocsparse_float_complex*)du,
                                               (rocsparse_float_complex*)x,
                                               batchCount,
                                               batchStride,
                                               pBuffer));
}

hipsparseStatus_t hipsparseZgtsv2StridedBatch(hipsparseHandle_t       handle,
                                              int                     m,
                                              const hipDoubleComplex* dl,
                                              const hipDoubleComplex* d,
                                              const hipDoubleComplex* du,
                                              hipDoubleComplex*       x,
                                              int                     batchCount,
                                              int                     batchStride,
                                              void*                   pBuffer)

{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_zgtsv_no_pivot_strided_batch((rocsparse_handle)handle,
                                               m,
                                               (const rocsparse_double_complex*)dl,
                                               (const rocsparse_double_complex*)d,
                                               (const rocsparse_double_complex*)du,
                                               (rocsparse_double_complex*)x,
                                               batchCount,
                                               batchStride,
                                               pBuffer));
}

hipsparseStatus_t hipsparseSgtsv2_bufferSizeExt(hipsparseHandle_t handle,
                                                int               m,
                                                int               n,
                                                const float*      dl,
                                                const float*      d,
                                                const float*      du,
                                                const float*      B,
                                                int               ldb,
                                                size_t*           pBufferSizeInBytes)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_sgtsv_buffer_size(
        (rocsparse_handle)handle, m, n, dl, d, du, B, ldb, pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseDgtsv2_bufferSizeExt(hipsparseHandle_t handle,
                                                int               m,
                                                int               n,
                                                const double*     dl,
                                                const double*     d,
                                                const double*     du,
                                                const double*     B,
                                                int               ldb,
                                                size_t*           pBufferSizeInBytes)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_dgtsv_buffer_size(
        (rocsparse_handle)handle, m, n, dl, d, du, B, ldb, pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseCgtsv2_bufferSizeExt(hipsparseHandle_t handle,
                                                int               m,
                                                int               n,
                                                const hipComplex* dl,
                                                const hipComplex* d,
                                                const hipComplex* du,
                                                const hipComplex* B,
                                                int               ldb,
                                                size_t*           pBufferSizeInBytes)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_cgtsv_buffer_size((rocsparse_handle)handle,
                                    m,
                                    n,
                                    (const rocsparse_float_complex*)dl,
                                    (const rocsparse_float_complex*)d,
                                    (const rocsparse_float_complex*)du,
                                    (const rocsparse_float_complex*)B,
                                    ldb,
                                    pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseZgtsv2_bufferSizeExt(hipsparseHandle_t       handle,
                                                int                     m,
                                                int                     n,
                                                const hipDoubleComplex* dl,
                                                const hipDoubleComplex* d,
                                                const hipDoubleComplex* du,
                                                const hipDoubleComplex* B,
                                                int                     ldb,
                                                size_t*                 pBufferSizeInBytes)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_zgtsv_buffer_size((rocsparse_handle)handle,
                                    m,
                                    n,
                                    (const rocsparse_double_complex*)dl,
                                    (const rocsparse_double_complex*)d,
                                    (const rocsparse_double_complex*)du,
                                    (const rocsparse_double_complex*)B,
                                    ldb,
                                    pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseSgtsv2(hipsparseHandle_t handle,
                                  int               m,
                                  int               n,
                                  const float*      dl,
                                  const float*      d,
                                  const float*      du,
                                  float*            B,
                                  int               ldb,
                                  void*             pBuffer)

{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_sgtsv((rocsparse_handle)handle, m, n, dl, d, du, B, ldb, pBuffer));
}

hipsparseStatus_t hipsparseDgtsv2(hipsparseHandle_t handle,
                                  int               m,
                                  int               n,
                                  const double*     dl,
                                  const double*     d,
                                  const double*     du,
                                  double*           B,
                                  int               ldb,
                                  void*             pBuffer)

{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_dgtsv((rocsparse_handle)handle, m, n, dl, d, du, B, ldb, pBuffer));
}

hipsparseStatus_t hipsparseCgtsv2(hipsparseHandle_t handle,
                                  int               m,
                                  int               n,
                                  const hipComplex* dl,
                                  const hipComplex* d,
                                  const hipComplex* du,
                                  hipComplex*       B,
                                  int               ldb,
                                  void*             pBuffer)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_cgtsv((rocsparse_handle)handle,
                                                      m,
                                                      n,
                                                      (const rocsparse_float_complex*)dl,
                                                      (const rocsparse_float_complex*)d,
                                                      (const rocsparse_float_complex*)du,
                                                      (rocsparse_float_complex*)B,
                                                      ldb,
                                                      pBuffer));
}

hipsparseStatus_t hipsparseZgtsv2(hipsparseHandle_t       handle,
                                  int                     m,
                                  int                     n,
                                  const hipDoubleComplex* dl,
                                  const hipDoubleComplex* d,
                                  const hipDoubleComplex* du,
                                  hipDoubleComplex*       B,
                                  int                     ldb,
                                  void*                   pBuffer)

{
    return rocSPARSEStatusToHIPStatus(rocsparse_zgtsv((rocsparse_handle)handle,
                                                      m,
                                                      n,
                                                      (const rocsparse_double_complex*)dl,
                                                      (const rocsparse_double_complex*)d,
                                                      (const rocsparse_double_complex*)du,
                                                      (rocsparse_double_complex*)B,
                                                      ldb,
                                                      pBuffer));
}

hipsparseStatus_t hipsparseSgtsv2_nopivot_bufferSizeExt(hipsparseHandle_t handle,
                                                        int               m,
                                                        int               n,
                                                        const float*      dl,
                                                        const float*      d,
                                                        const float*      du,
                                                        const float*      B,
                                                        int               ldb,
                                                        size_t*           pBufferSizeInBytes)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_sgtsv_no_pivot_buffer_size(
        (rocsparse_handle)handle, m, n, dl, d, du, B, ldb, pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseDgtsv2_nopivot_bufferSizeExt(hipsparseHandle_t handle,
                                                        int               m,
                                                        int               n,
                                                        const double*     dl,
                                                        const double*     d,
                                                        const double*     du,
                                                        const double*     B,
                                                        int               ldb,
                                                        size_t*           pBufferSizeInBytes)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_dgtsv_no_pivot_buffer_size(
        (rocsparse_handle)handle, m, n, dl, d, du, B, ldb, pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseCgtsv2_nopivot_bufferSizeExt(hipsparseHandle_t handle,
                                                        int               m,
                                                        int               n,
                                                        const hipComplex* dl,
                                                        const hipComplex* d,
                                                        const hipComplex* du,
                                                        const hipComplex* B,
                                                        int               ldb,
                                                        size_t*           pBufferSizeInBytes)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_cgtsv_no_pivot_buffer_size((rocsparse_handle)handle,
                                             m,
                                             n,
                                             (const rocsparse_float_complex*)dl,
                                             (const rocsparse_float_complex*)d,
                                             (const rocsparse_float_complex*)du,
                                             (const rocsparse_float_complex*)B,
                                             ldb,
                                             pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseZgtsv2_nopivot_bufferSizeExt(hipsparseHandle_t       handle,
                                                        int                     m,
                                                        int                     n,
                                                        const hipDoubleComplex* dl,
                                                        const hipDoubleComplex* d,
                                                        const hipDoubleComplex* du,
                                                        const hipDoubleComplex* B,
                                                        int                     ldb,
                                                        size_t*                 pBufferSizeInBytes)
{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_zgtsv_no_pivot_buffer_size((rocsparse_handle)handle,
                                             m,
                                             n,
                                             (const rocsparse_double_complex*)dl,
                                             (const rocsparse_double_complex*)d,
                                             (const rocsparse_double_complex*)du,
                                             (const rocsparse_double_complex*)B,
                                             ldb,
                                             pBufferSizeInBytes));
}

hipsparseStatus_t hipsparseSgtsv2_nopivot(hipsparseHandle_t handle,
                                          int               m,
                                          int               n,
                                          const float*      dl,
                                          const float*      d,
                                          const float*      du,
                                          float*            B,
                                          int               ldb,
                                          void*             pBuffer)

{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_sgtsv_no_pivot((rocsparse_handle)handle, m, n, dl, d, du, B, ldb, pBuffer));
}

hipsparseStatus_t hipsparseDgtsv2_nopivot(hipsparseHandle_t handle,
                                          int               m,
                                          int               n,
                                          const double*     dl,
                                          const double*     d,
                                          const double*     du,
                                          double*           B,
                                          int               ldb,
                                          void*             pBuffer)

{
    return rocSPARSEStatusToHIPStatus(
        rocsparse_dgtsv_no_pivot((rocsparse_handle)handle, m, n, dl, d, du, B, ldb, pBuffer));
}

hipsparseStatus_t hipsparseCgtsv2_nopivot(hipsparseHandle_t handle,
                                          int               m,
                                          int               n,
                                          const hipComplex* dl,
                                          const hipComplex* d,
                                          const hipComplex* du,
                                          hipComplex*       B,
                                          int               ldb,
                                          void*             pBuffer)

{
    return rocSPARSEStatusToHIPStatus(rocsparse_cgtsv_no_pivot((rocsparse_handle)handle,
                                                               m,
                                                               n,
                                                               (const rocsparse_float_complex*)dl,
                                                               (const rocsparse_float_complex*)d,
                                                               (const rocsparse_float_complex*)du,
                                                               (rocsparse_float_complex*)B,
                                                               ldb,
                                                               pBuffer));
}

hipsparseStatus_t hipsparseZgtsv2_nopivot(hipsparseHandle_t       handle,
                                          int                     m,
                                          int                     n,
                                          const hipDoubleComplex* dl,
                                          const hipDoubleComplex* d,
                                          const hipDoubleComplex* du,
                                          hipDoubleComplex*       B,
                                          int                     ldb,
                                          void*                   pBuffer)

{
    return rocSPARSEStatusToHIPStatus(rocsparse_zgtsv_no_pivot((rocsparse_handle)handle,
                                                               m,
                                                               n,
                                                               (const rocsparse_double_complex*)dl,
                                                               (const rocsparse_double_complex*)d,
                                                               (const rocsparse_double_complex*)du,
                                                               (rocsparse_double_complex*)B,
                                                               ldb,
                                                               pBuffer));
}

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseScsrcolor(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       nnz,
                                     const hipsparseMatDescr_t descrA,
                                     const float*              csrValA,
                                     const int*                csrRowPtrA,
                                     const int*                csrColIndA,
                                     const float*              fractionToColor,
                                     int*                      ncolors,
                                     int*                      coloring,
                                     int*                      reordering,
                                     hipsparseColorInfo_t      info)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_scsrcolor((rocsparse_handle)handle,
                                                          m,
                                                          nnz,
                                                          (const rocsparse_mat_descr)descrA,
                                                          csrValA,
                                                          csrRowPtrA,
                                                          csrColIndA,
                                                          fractionToColor,
                                                          ncolors,
                                                          coloring,
                                                          reordering,
                                                          (rocsparse_mat_info)info));
}

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseDcsrcolor(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       nnz,
                                     const hipsparseMatDescr_t descrA,
                                     const double*             csrValA,
                                     const int*                csrRowPtrA,
                                     const int*                csrColIndA,
                                     const double*             fractionToColor,
                                     int*                      ncolors,
                                     int*                      coloring,
                                     int*                      reordering,
                                     hipsparseColorInfo_t      info)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_dcsrcolor((rocsparse_handle)handle,
                                                          m,
                                                          nnz,
                                                          (const rocsparse_mat_descr)descrA,
                                                          csrValA,
                                                          csrRowPtrA,
                                                          csrColIndA,
                                                          fractionToColor,
                                                          ncolors,
                                                          coloring,
                                                          reordering,
                                                          (rocsparse_mat_info)info));
}

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseCcsrcolor(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       nnz,
                                     const hipsparseMatDescr_t descrA,
                                     const hipComplex*         csrValA,
                                     const int*                csrRowPtrA,
                                     const int*                csrColIndA,
                                     const float*              fractionToColor,
                                     int*                      ncolors,
                                     int*                      coloring,
                                     int*                      reordering,
                                     hipsparseColorInfo_t      info)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_ccsrcolor((rocsparse_handle)handle,
                                                          m,
                                                          nnz,
                                                          (const rocsparse_mat_descr)descrA,
                                                          (const rocsparse_float_complex*)csrValA,
                                                          csrRowPtrA,
                                                          csrColIndA,
                                                          fractionToColor,
                                                          ncolors,
                                                          coloring,
                                                          reordering,
                                                          (rocsparse_mat_info)info));
}

HIPSPARSE_EXPORT
hipsparseStatus_t hipsparseZcsrcolor(hipsparseHandle_t         handle,
                                     int                       m,
                                     int                       nnz,
                                     const hipsparseMatDescr_t descrA,
                                     const hipDoubleComplex*   csrValA,
                                     const int*                csrRowPtrA,
                                     const int*                csrColIndA,
                                     const double*             fractionToColor,
                                     int*                      ncolors,
                                     int*                      coloring,
                                     int*                      reordering,
                                     hipsparseColorInfo_t      info)
{
    return rocSPARSEStatusToHIPStatus(rocsparse_zcsrcolor((rocsparse_handle)handle,
                                                          m,
                                                          nnz,
                                                          (const rocsparse_mat_descr)descrA,
                                                          (const rocsparse_double_complex*)csrValA,
                                                          csrRowPtrA,
                                                          csrColIndA,
                                                          fractionToColor,
                                                          ncolors,
                                                          coloring,
                                                          reordering,
                                                          (rocsparse_mat_info)info));
}

#ifdef __cplusplus
}
#endif
