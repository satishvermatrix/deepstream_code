/*
 * Custom nvinfer bbox parser for Ultralytics YOLO26 (and YOLO10+) end-to-end
 * ONNX output: [num_det, 6] = (x1, y1, x2, y2, conf, cls) in network pixels.
 * Pair with cluster-mode=4. Do not run DeepStream NMS on this tensor.
 */

#include "nvdsinfer_custom_impl.h"

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <vector>

static float clip(float v, float lo, float hi)
{
    return std::max(lo, std::min(v, hi));
}

static bool parseYolo26E2E(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector<NvDsInferObjectDetectionInfo> &objectList)
{
    if (outputLayersInfo.empty()) {
        std::cerr << "YOLO26 parser: no output layers\n";
        return false;
    }

    NvDsInferLayerInfo const &out = outputLayersInfo[0];
    NvDsInferDims const &dims = out.inferDims;

    static bool logged = false;
    if (!logged) {
        std::cout << "YOLO26 parser layer=" << out.layerName << " numDims=" << dims.numDims
                  << " d=[";
        for (int i = 0; i < dims.numDims; ++i) {
            std::cout << dims.d[i] << (i + 1 < dims.numDims ? "," : "");
        }
        std::cout << "] net=" << networkInfo.width << "x" << networkInfo.height << std::endl;
        logged = true;
    }

    int numDet = 0;
    int stride = 6;
    if (dims.numDims == 2 && dims.d[1] == 6) {
        numDet = static_cast<int>(dims.d[0]);
    } else if (dims.numDims == 2 && dims.d[0] == 6) {
        std::cerr << "YOLO26 parser: unexpected transposed output [6, N]\n";
        return false;
    } else if (dims.numDims == 3 && dims.d[2] == 6) {
        numDet = static_cast<int>(dims.d[1]);
    } else if (dims.numDims == 3 && dims.d[1] == 6) {
        std::cerr << "YOLO26 parser: unexpected layout [B, 6, N]\n";
        return false;
    } else {
        std::cerr << "YOLO26 parser: expected post-NMS [N,6] or [1,N,6]\n";
        return false;
    }

    auto const *data = static_cast<float const *>(out.buffer);
    if (!data) {
        return false;
    }

    float const netW = static_cast<float>(networkInfo.width);
    float const netH = static_cast<float>(networkInfo.height);
    int const ncls = static_cast<int>(detectionParams.numClassesConfigured);

    objectList.clear();
    for (int i = 0; i < numDet; ++i) {
        float const *row = data + i * stride;
        float conf = row[4];
        int cls = static_cast<int>(row[5]);
        if (cls < 0 || cls >= ncls) {
            continue;
        }
        if (conf < detectionParams.perClassPreclusterThreshold[cls]) {
            continue;
        }

        float x1 = clip(row[0], 0.f, netW);
        float y1 = clip(row[1], 0.f, netH);
        float x2 = clip(row[2], 0.f, netW);
        float y2 = clip(row[3], 0.f, netH);
        float w = x2 - x1;
        float h = y2 - y1;
        if (w < 1.f || h < 1.f) {
            continue;
        }

        NvDsInferObjectDetectionInfo obj{};
        obj.classId = static_cast<unsigned int>(cls);
        obj.left = x1;
        obj.top = y1;
        obj.width = w;
        obj.height = h;
        obj.detectionConfidence = conf;
        objectList.push_back(obj);
    }
    return true;
}

extern "C" bool NvDsInferParseYolo26(
    std::vector<NvDsInferLayerInfo> const &outputLayersInfo,
    NvDsInferNetworkInfo const &networkInfo,
    NvDsInferParseDetectionParams const &detectionParams,
    std::vector<NvDsInferObjectDetectionInfo> &objectList)
{
    return parseYolo26E2E(outputLayersInfo, networkInfo, detectionParams, objectList);
}

CHECK_CUSTOM_PARSE_FUNC_PROTOTYPE(NvDsInferParseYolo26);
