vulkano_shaders::shader! {
    ty: "compute",
    linalg_type: "nalgebra",
    vulkan_version: "1.2",
    spirv_version: "1.3",
    src: r"
#version 460 core

#extension GL_EXT_buffer_reference : require
#extension GL_KHR_shader_subgroup_basic: enable
#extension GL_KHR_shader_subgroup_arithmetic: enable
#extension GL_KHR_shader_subgroup_ballot: enable

const int RADIX = 256;
#define WORKGROUP_SIZE 512
#define PARTITION_DIVISION 8
const int PARTITION_SIZE = PARTITION_DIVISION * WORKGROUP_SIZE;

layout (local_size_x = WORKGROUP_SIZE) in;

layout (buffer_reference, std430) readonly buffer ElementCount {
  uint elementCount;
};

layout (buffer_reference, std430) readonly buffer GlobalHistogram {
  uint globalHistogram[];  // (4, R)
};

layout (buffer_reference, std430) readonly buffer PartitionHistogram {
  uint partitionHistogram[];  // (P, R)
};

layout (buffer_reference, std430) buffer Keys {
  uint keys[];  // (N)
};

#ifdef KEY_VALUE
layout (buffer_reference, std430) buffer Values {
  uint values[];  // (N)
};
#endif

layout (push_constant) uniform PushConstant {
  int pass;
  restrict ElementCount elementCountReference;
  restrict GlobalHistogram globalHistogramReference;
  restrict PartitionHistogram partitionHistogramReference;
  restrict Keys keysInReference;
  restrict Keys keysOutReference;
#ifdef KEY_VALUE
  restrict Values valuesInReference;
  restrict Values valuesOutReference;
#endif
};

const uint SHMEM_SIZE = PARTITION_SIZE;

shared uint localHistogram[SHMEM_SIZE];  // (R, S=16)=4096, (P) for alias. take maximum.
shared uint localHistogramSum[RADIX];

// returns 0b00000....11111, where msb is id-1.
uvec4 GetExclusiveSubgroupMask(uint id) {
  // clamp bit-shift right operand between 0..31 to avoid undefined behavior.
  uint shift = (1 << bitfieldExtract(id, 0, 5)) - 1;  //  (1 << (id % 32)) - 1
  // right shift operation on signed integer copies sign bit, use the trick for masking.
  // (negative)     >> 31 = 111...111
  // (non-negative) >> 31 = 000...000
  int x = int(id) >> 5;
  return uvec4(
    (shift & ((-1-x) >> 31)) | ((0-x) >> 31),
    (shift & (( 0-x) >> 31)) | ((1-x) >> 31),
    (shift & (( 1-x) >> 31)) | ((2-x) >> 31),
    (shift & (( 2-x) >> 31)) | ((3-x) >> 31)
  );
}

uint GetBitCount(uvec4 value) {
  uvec4 result = bitCount(value);
  return result[0] + result[1] + result[2] + result[3];
}

void main() {
  uint threadIndex = gl_SubgroupInvocationID;  // 0..31 or 0..63
  uint subgroupIndex = gl_SubgroupID;  // 0..15 or 0..7
  uint index = subgroupIndex * gl_SubgroupSize + threadIndex;
  uvec4 subgroupMask = GetExclusiveSubgroupMask(threadIndex);

  uint partitionIndex = gl_WorkGroupID.x;
  uint partitionStart = partitionIndex * PARTITION_SIZE;

  uint elementCount = elementCountReference.elementCount;

  if (partitionStart >= elementCount) return;

  if (index < RADIX) {
    for (int i = 0; i < gl_NumSubgroups; ++i) {
      localHistogram[gl_NumSubgroups * index + i] = 0;
    }
  }
  barrier();

  // load from global memory, local histogram and offset
  uint localKeys[PARTITION_DIVISION];
  uint localRadix[PARTITION_DIVISION];
  uint localOffsets[PARTITION_DIVISION];
  uint subgroupHistogram[PARTITION_DIVISION];

#ifdef KEY_VALUE
  uint localValues[PARTITION_DIVISION];
#endif
  for (int i = 0; i < PARTITION_DIVISION; ++i) {
    uint keyIndex = partitionStart + (PARTITION_DIVISION * gl_SubgroupSize) * subgroupIndex + i * gl_SubgroupSize + threadIndex;
    uint key = keyIndex < elementCount ? keysInReference.keys[keyIndex] : 0xffffffff;
    localKeys[i] = key;
    
#ifdef KEY_VALUE
    localValues[i] = keyIndex < elementCount ? valuesInReference.values[keyIndex] : 0;
#endif

    uint radix = bitfieldExtract(key, pass * 8, 8);
    localRadix[i] = radix;

    // mask per digit
    uvec4 mask = subgroupBallot(true);
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
      uint digit = (radix >> j) & 1;
      uvec4 ballot = subgroupBallot(digit == 1);
      // digit - 1 is 0 or 0xffffffff. xor to flip.
      mask &= uvec4(digit - 1) ^ ballot;
    }

    // subgroup level offset for radix
    uint subgroupOffset = GetBitCount(subgroupMask & mask);
    uint radixCount = GetBitCount(mask);

    // elect a representative per radix, add to histogram
    if (subgroupOffset == 0) {
      // accumulate to local histogram
      atomicAdd(localHistogram[gl_NumSubgroups * radix + subgroupIndex], radixCount);
      subgroupHistogram[i] = radixCount;
    } else {
      subgroupHistogram[i] = 0;
    }

    localOffsets[i] = subgroupOffset;
  }
  barrier();
  
  // local histogram reduce 4096 or 2048
  for (uint i = index; i < RADIX * gl_NumSubgroups; i += WORKGROUP_SIZE) {
    uint v = localHistogram[i];
    uint sum = subgroupAdd(v);
    uint excl = subgroupExclusiveAdd(v);
    localHistogram[i] = excl;
    if (threadIndex == 0) {
      localHistogramSum[i / gl_SubgroupSize] = sum;
    }
  }
  barrier();

  // local histogram reduce 128 or 32
  uint intermediateOffset0 = RADIX * gl_NumSubgroups / gl_SubgroupSize;
  if (index < intermediateOffset0) {
    uint v = localHistogramSum[index];
    uint sum = subgroupAdd(v);
    uint excl = subgroupExclusiveAdd(v);
    localHistogramSum[index] = excl;
    if (threadIndex == 0) {
      localHistogramSum[intermediateOffset0 + index / gl_SubgroupSize] = sum;
    }
  }
  barrier();

  // local histogram reduce 4 or 1
  uint intermediateSize1 = max(RADIX * gl_NumSubgroups / gl_SubgroupSize / gl_SubgroupSize, 1);
  if (index < intermediateSize1) {
    uint v = localHistogramSum[intermediateOffset0 + index];
    uint excl = subgroupExclusiveAdd(v);
    localHistogramSum[intermediateOffset0 + index] = excl;
  }
  barrier();

  // local histogram add 128
  if (index < intermediateOffset0) {
    localHistogramSum[index] += localHistogramSum[intermediateOffset0 + index / gl_SubgroupSize];
  }
  barrier();

  // local histogram add 4096
  for (uint i = index; i < RADIX * gl_NumSubgroups; i += WORKGROUP_SIZE) {
    localHistogram[i] += localHistogramSum[i / gl_SubgroupSize];
  }
  barrier();

  // post-scan stage
  for (int i = 0; i < PARTITION_DIVISION; ++i) {
    uint radix = localRadix[i];
    localOffsets[i] += localHistogram[gl_NumSubgroups * radix + subgroupIndex];

    barrier();
    if (subgroupHistogram[i] > 0) {
      atomicAdd(localHistogram[gl_NumSubgroups * radix + subgroupIndex], subgroupHistogram[i]);
    }
    barrier();
  }

  // after atomicAdd, localHistogram contains inclusive sum
  if (index < RADIX) {
    uint v = index == 0 ? 0 : localHistogram[gl_NumSubgroups * index - 1];
    localHistogramSum[index] = globalHistogramReference.globalHistogram[RADIX * pass + index] + partitionHistogramReference.partitionHistogram[RADIX * partitionIndex + index] - v;
  }
  barrier();

  // rearrange keys. grouping keys together makes dstOffset to be almost sequential, grants huge speed boost.
  // now localHistogram is unused, so alias memory.
  for (int i = 0; i < PARTITION_DIVISION; ++i) {
    localHistogram[localOffsets[i]] = localKeys[i];
  }
  barrier();

  // binning
  for (uint i = index; i < PARTITION_SIZE; i += WORKGROUP_SIZE) {
    uint key = localHistogram[i];
    uint radix = bitfieldExtract(key, pass * 8, 8);
    uint dstOffset = localHistogramSum[radix] + i;
    if (dstOffset < elementCount) {
      keysOutReference.keys[dstOffset] = key;
    }

#ifdef KEY_VALUE
    localKeys[i / WORKGROUP_SIZE] = dstOffset;
#endif
  }

#ifdef KEY_VALUE
  barrier();

  for (int i = 0; i < PARTITION_DIVISION; ++i) {
    localHistogram[localOffsets[i]] = localValues[i];
  }
  barrier();

  for (uint i = index; i < PARTITION_SIZE; i += WORKGROUP_SIZE) {
    uint value = localHistogram[i];
    uint dstOffset = localKeys[i / WORKGROUP_SIZE];
    if (dstOffset < elementCount) {
      valuesOutReference.values[dstOffset] = value;
    }
  }
#endif
}
"
}