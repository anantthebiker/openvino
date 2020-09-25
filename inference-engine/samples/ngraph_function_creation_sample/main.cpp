// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <format_reader_ptr.h>
#include <gflags/gflags.h>

#include <inference_engine.hpp>
#include <limits>
#include <memory>
#include <samples/args_helper.hpp>
#include <samples/common.hpp>
#include <samples/slog.hpp>
#include <samples/classification_results.h>
#include <string>
#include <vector>

#include "ngraph_function_creation_sample.hpp"
#include "ngraph/ngraph.hpp"
#include "gna/gna_config.hpp"
#include "include/ngraph_ops/fully_connected.hpp"

using namespace InferenceEngine;
using namespace ngraph;

bool ParseAndCheckCommandLine(int argc, char* argv[]) {
    slog::info << "Parsing input parameters" << slog::endl;

    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    if (FLAGS_h) {
        showUsage();
        showAvailableDevices();
        return false;
    }

    if (FLAGS_nt <= 0 || FLAGS_nt > 10) {
        throw std::logic_error("Incorrect value for nt argument. It should be greater than 0 and less than 10.");
    }

    return true;
}

void readFile(const std::string& file_name, void* buffer, size_t maxSize) {
    std::ifstream inputFile;

    inputFile.open(file_name, std::ios::binary | std::ios::in);
    if (!inputFile.is_open()) {
        throw std::logic_error("Cannot open weights file");
    }

    if (!inputFile.read(reinterpret_cast<char*>(buffer), maxSize)) {
        inputFile.close();
        throw std::logic_error("Cannot read bytes from weights file");
    }

    inputFile.close();
}

TBlob<uint8_t>::CPtr ReadWeights(std::string filepath) {
    std::ifstream weightFile(filepath, std::ifstream::ate | std::ifstream::binary);
    int64_t fileSize = weightFile.tellg();

    if (fileSize < 0) {
        throw std::logic_error("Incorrect weights file");
    }

    size_t ulFileSize = static_cast<size_t>(fileSize);

    TBlob<uint8_t>::Ptr weightsPtr(new TBlob<uint8_t>({Precision::FP32, {ulFileSize}, Layout::C}));
    weightsPtr->allocate();
    readFile(filepath, weightsPtr->buffer(), ulFileSize);

    return weightsPtr;
}

std::shared_ptr<Function> createNgraphFunction() {
    // std::vector<uint32_t> values_input_x = {1, 2, 3, 4, 5, 6};
    auto input_x = std::make_shared<op::Parameter>(element::Type_t::f32, Shape{6, 1});
    std::vector<uint32_t> values_input_h = {1, 2, 3, 4, 5, 6};
    std::shared_ptr<Node> input_h = std::make_shared<op::Constant>(element::Type_t::f32, Shape{6, 1}, values_input_h);
    // std::vector< std::vector<int32_t>> values_weight_forgetGate_x = {{245, 42, 218,  84},
    //                                                     {122,  56, 179, 185},
    //                                                     {209, 101,  85, 218},
    //                                                     {145, 121, 204, 172},
    //                                                     {187, 247,  21, 251},
    //                                                     {154,  11, 127, 146}};
    std::vector< uint32_t> values_weight_forgetGate_x = {1, 2, 3, 4, 5, 6 };
                                                        // 1, 2, 3, 4, 5, 6,
                                                        // 1, 2, 3, 4, 5, 6,
                                                        // 1, 2, 3, 4, 5, 6}; //Its a row major vector of Constants.
    std::vector< uint32_t> values_weight_forgetGate_h = {1, 2, 3, 4, 5, 6, \
                                                        1, 2, 3, 4, 5, 6, \
                                                        1, 2, 3, 4, 5, 6, \
                                                        1, 2, 3, 4, 5, 6};
    //With 4 neurons & 4X6 connections to input layer.
    std::shared_ptr<Node> weight_forgetGate_x = std::make_shared<op::Constant>(element::Type_t::f32, Shape{1, 6}, values_weight_forgetGate_x);
    //With 4 neurons & 4X6 connections to previous state layer.
    std::shared_ptr<Node> weight_forgetGate_h = std::make_shared<op::Constant>(element::Type_t::f32, Shape{4, 6}, values_weight_forgetGate_h);
    //std::vector<int32_t> values_bias_forgetGate = {1, 1, 1, 1};
    std::vector<int32_t> values_bias_forgetGate = { 1, 1, 1, 1, 1, 1,
                                                    1, 1, 1, 1, 1, 1,
                                                    1, 1, 1, 1, 1, 1,
                                                    1, 1, 1, 1, 1, 1,
                                                    1, 1, 1, 1, 1, 1,
                                                    1, 1, 1, 1, 1, 1};
    std::shared_ptr<Node> bias_forgetGate = std::make_shared<op::Constant>(element::Type_t::f32, Shape{6, 6}, values_bias_forgetGate); //4 ,1 Earlier
    std::vector<int32_t> values_zero = {0, 0, 0, 0};
    std::shared_ptr<Node> bias_zero_matrix = std::make_shared<op::Constant>(element::Type_t::f32, Shape{4, 1}, values_zero);
    std::vector<int32_t> values_identity = {1};
    std::shared_ptr<Node> identity_matrix = std::make_shared<op::Constant>(element::Type_t::f32, Shape{1, 1}, values_identity);
    /*********************
    Matrix Multiplication:
    *********************/
    //auto matMul_W_X_reshape = std::make_shared<op::Broadcast>(matMul_weightX_inpX, Shape{4, 1}, AxisSet{0});
    //std::shared_ptr<Node> matmul_identity_result = std::make_shared<op::MatMul>(input_x, identity_matrix);
    //std::vector<int32_t> values_matmul_identity_result = matmul_identity_result->
    //std::shared_ptr<Node> matmul_identity_result_constant = std::make_shared<op::Constant>(element::Type_t::f32, Shape{6, 1}, *matmul_identity_result);
    //auto matMul_weightX_inpX = std::make_shared<op::MatMul>(weight_forgetGate_x, matmul_identity_result_constant);
    auto matMul_weightX_inpX = std::make_shared<op::MatMul>(input_x, weight_forgetGate_x);
    //auto matMul_weightX_inpX = std::make_shared<op::FullyConnected>(weight_forgetGate_x, input_x, bias_zero_matrix, Shape{4, 1});
    //auto matMul_weightH_inpH = std::make_shared<op::MatMul>(weight_forgetGate_h, input_h);
    //auto matMul_weightH_inpH = std::make_shared<op::FullyConnected>(weight_forgetGate_h, input_x, bias_zero_matrix, Shape{4, 1});
    //auto matMul_W_H_reshape = std::make_shared<op::Broadcast>(matMul_weightH_inpH, Shape{4, 1}, AxisSet{0});
    //?? //Broadcasting Bias(4X1) to a 4X1 along column axis.
    // std::shared_ptr<Node> bias_forgetGate_broadcast = std::make_shared<op::Broadcast>(bias_forgetGate, Shape{4, 1}, AxisSet{1});
    //output_forgetGate of type Node Therefore can be given as result.
    //auto output_forgetGate = matMul_weightX_inpX + matMul_weightH_inpH + bias_forgetGate_broadcast;
    auto output_forgetGate =   matMul_weightX_inpX + bias_forgetGate; //+ matMul_weightX_inpX
    // auto output_forgetGate = std::make_shared<op::Sigmoid>(matMul_weightX_inpX + matMul_weightH_inpH + bias_forgetGate_broadcast);
    std::shared_ptr<ngraph::Function> fnPtr = std::make_shared<ngraph::Function>(OutputVector{ output_forgetGate }, ngraph::ParameterVector{ input_x }, "lstm");
    return fnPtr;
}

/**
 * @brief The entry point for inference engine automatic ngraph function creation sample
 * @file ngraph_function_creation_sample/main.cpp
 * @example ngraph_function_creation_sample/main.cpp
 */
int main(int argc, char* argv[]) {
    std::cout << "Hello Anant" << std::endl;
    try {
        std::string scale_factor = "1";
        slog::info << "InferenceEngine: " << GetInferenceEngineVersion() << slog::endl;

        if (!ParseAndCheckCommandLine(argc, argv)) {
            return 0;
        }

        // /** This vector stores paths to the processed images **/
        // std::vector<std::string> images;
        // parseInputFilesArguments(images);
        // if (images.empty()) {
        //     throw std::logic_error("No suitable images were found");
        // }

        // --------------------------- 1. Load inference engine -------------------------------------
        slog::info << "Loading Inference Engine" << slog::endl;
        Core ie;

        slog::info << "Device info: " << slog::endl;
        std::cout << ie.GetVersions(FLAGS_d) << std::endl;
        // -----------------------------------------------------------------------------------------------------

        //--------------------------- 2. Create network using ngraph function -----------------------------------

        CNNNetwork network(createNgraphFunction());
        InferenceEngine::ResponseDesc desc;
        std::cout << "Created" << std::endl;
        // network.serialize("/tmp/network.xml", "/tmp/network.bin");
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 3. Configure input & output ---------------------------------------------
        // --------------------------- Prepare input blobs -----------------------------------------------------
        slog::info << "Preparing input blobs" << slog::endl;

        InputsDataMap inputInfo = network.getInputsInfo();
        if (inputInfo.size() != 1) {
            throw std::logic_error("Sample supports topologies only with 1 input");
        }

        auto inputInfoItem = *inputInfo.begin();

        /** Specifying the precision and layout of input data provided by the user.
         * Call this before loading the network to the device **/
        inputInfoItem.second->setPrecision(Precision::FP32);
        inputInfoItem.second->setLayout(Layout::NC); //NC is for 2D.
        /*
        ********************
        Network config file:
        ********************
        */
        std::map<std::string, std::string> config;
        std::map<std::string, std::string> gnaPluginConfig;
        gnaPluginConfig[InferenceEngine::GNAConfigParams::KEY_GNA_DEVICE_MODE] = "GNA_SW_FP32"; //SW_FP32
        gnaPluginConfig[InferenceEngine::GNAConfigParams::KEY_GNA_PRECISION] = "I16";

        std::string scaleFactorConfigKey_1 = GNA_CONFIG_KEY(SCALE_FACTOR) + std::string("_") + std::to_string(0);
        std::string scaleFactorConfigKey_2 = GNA_CONFIG_KEY(SCALE_FACTOR) + std::string("_") + std::to_string(1);
        gnaPluginConfig[scaleFactorConfigKey_1] = scale_factor;
        gnaPluginConfig[scaleFactorConfigKey_2] = "32766";
        gnaPluginConfig[GNA_CONFIG_KEY(COMPACT_MODE)] = CONFIG_VALUE(NO);
        config.insert(std::begin(gnaPluginConfig), std::end(gnaPluginConfig));

        auto executable_network = ie.LoadNetwork(network, "CPU"); //CPU or FLAGS_d? //(network, "GNA", config) OR (network, "CPU")
        std::cout << "Have comeafter executableNetwork" << std::endl;
        auto inferRequest = executable_network.CreateInferRequest();

        std::cout << "Network name = " << network.getName() << "\n";
        std::vector<InferenceEngine::Blob::Ptr> ptrInputBlobs;
        for (auto& input : network.getInputsInfo()) {
            //VLOG(L1,"%p", (void*)enginePtr->inferRequest.GetBlob(input.first));
            ptrInputBlobs.push_back(inferRequest.GetBlob(input.first));
            std::cout << "Input name = " << input.first << "\n";
        }
        //Q: Why ptrInputBlobs[0]->byteSize()/4 ??
        std::cout << "ptrInBlob = " << ptrInputBlobs[0] << " size = " << ptrInputBlobs.size() << " " << ptrInputBlobs[0]->byteSize()/4 << std::endl;
        // std::vector<std::vector<float>> image_data;
        std::vector<float> values_input_x = {6, 5, 4, 3, 2, 1};
        float* dest = ptrInputBlobs[0]->buffer().as<float*>(); //as an U32
        // // int i = 0;
        // // int j =0;
        // std::ifstream input;
        // std::vector<float> input_vals(cellSize);
        // input.open("/data/local/tmp/ip_to_ln.csv");
        // if(input.is_open())
        // // {
        // for (int i = 0; i < ptrInputBlobs[0]->byteSize()/4; i++) {
        //     input >> input_vals[i];
        //     input.get();
        // //        //image_data[j][k];
        // }
        //}
        // byteSize()/4 should tell how many UInt32 values can the ptrInputBlobs(input vector of size: 6X1 can take in)
        for (unsigned int j = 0; j < ptrInputBlobs[0]->byteSize()/4 ; j++) {
                *(dest + j) = values_input_x[j];
        //y_values << input_vals[j] << ",";
        }
        if (ptrInputBlobs.size() > 1) {
            throw std::logic_error("Sample supports topologies only with 1 input");
        // float* dest2 = ptrInputBlobs[1]->buffer().as<float*>();
        // for (j = 0; j < ptrInputBlobs[1]->byteSize()/4 ; j++) {
        //     *(dest2 + j) = -0.5;
        // }
        }
        //Does inference >>
        for (unsigned int i = 0; i < 1; i++) {
            inferRequest.StartAsync();  //for async infer
            inferRequest.Wait(200); //check right value to infer
        }

        InferenceEngine::OutputsDataMap outputInfo = network.getOutputsInfo();
        auto outputInfoItem = outputInfo.begin()->second;
        std::cout << "Inference done" <<  "\n";
        auto outputBlob = inferRequest.GetBlob(outputInfo.begin()->first);
        std::cout << "OPutputblob created" <<  "\n";
        float *op = outputBlob->buffer().as<float*>();
        std::cout << "Intel LN  values = " <<  outputBlob->byteSize()/4 << "\n";
        for (unsigned int i = 0; i < outputBlob->byteSize()/4; i++) {
            std::cout << *(op + i) << "\t";
            if (i % 6 == 5)
                std::cout << "\n";
        }
        std::cout << std::endl;
    //     std::vector<std::shared_ptr<unsigned char>> imagesData;
    //     for (auto& i : images) {
    //         FormatReader::ReaderPtr reader(i.c_str());
    //         if (reader.get() == nullptr) {
    //             slog::warn << "Image " + i + " cannot be read!" << slog::endl;
    //             continue;
    //         }
    //         /** Store image data **/
    //         std::shared_ptr<unsigned char> data(reader->getData(inputInfoItem.second->getTensorDesc().getDims()[3],
    //                                                             inputInfoItem.second->getTensorDesc().getDims()[2]));
    //         if (data.get() != nullptr) {
    //             imagesData.push_back(data);
    //         }
    //     }

    //     // if (imagesData.empty()) {
    //     //     throw std::logic_error("Valid input images were not found");
    //     // }

    //     /** Setting batch size using image count **/
    //     size_t batchSize = 1;
    //     slog::info << "Batch size is " << std::to_string(batchSize) << slog::endl;

    //     // --------------------------- Prepare output blobs -----------------------------------------------------
    //     slog::info << "Checking that the outputs are as the demo expects" << slog::endl;
    //     OutputsDataMap outputInfo(network.getOutputsInfo());
    //     std::string firstOutputName;

    //     for (auto& item : outputInfo) {
    //         if (firstOutputName.empty()) {
    //             firstOutputName = item.first;
    //         }
    //         DataPtr outputData = item.second;
    //         if (!outputData) {
    //             throw std::logic_error("Output data pointer is not valid");
    //         }

    //         item.second->setPrecision(Precision::FP32);
    //     }

    //     if (outputInfo.size() != 1) {
    //         throw std::logic_error("This demo accepts networks with a single output");
    //     }

    //     DataPtr& output = outputInfo.begin()->second;
    //     auto outputName = outputInfo.begin()->first;

    //     const SizeVector outputDims = output->getTensorDesc().getDims();
    //     const int classCount = outputDims[1]; Preparing input blobs

    //     if (classCount > 10) {
    //         throw std::logic_error("Incorrect number of output classes for LeNet network");
    //     }

    //     if (outputDims.size() != 2) {
    //         throw std::logic_error("Incorrect output dimensions for LeNet");
    //     }
    //     output->setPrecision(Precision::FP32);
    //     output->setLayout(Layout::NC);

    //     // -----------------------------------------------------------------------------------------------------

    //     // --------------------------- 4. Loading model to the device ------------------------------------------
    //     slog::info << "Loading model to the device" << slog::endl;
    //     ExecutableNetwork exeNetwork = ie.LoadNetwork(network, FLAGS_d);
    //     // -----------------------------------------------------------------------------------------------------

    //     // --------------------------- 5. Create infer request -------------------------------------------------
    //     slog::info << "Create infer request" << slog::endl;
    //     InferRequest infer_request = exeNetwork.CreateInferRequest();
    //     // -----------------------------------------------------------------------------------------------------

    //     // --------------------------- 6. Prepare input --------------------------------------------------------
    //     /** Iterate over all the input blobs **/
    //     for (const auto& item : inputInfo) {
    //         /** Creating input blob **/
    //         Blob::Ptr input = infer_request.GetBlob(item.first);

    //         /** Filling input tensor with images. First b channel, then g and r channels **/
    //         size_t num_channels = input->getTensorDesc().getDims()[1];
    //         size_t image_size = input->getTensorDesc().getDims()[2] * input->getTensorDesc().getDims()[3];

    //         auto data = input->buffer().as<PrecisionTrait<Precision::FP32>::value_type*>();

    //         /** Iterate over all input images **/
    //         for (size_t image_id = 0; image_id < imagesData.size(); ++image_id) {
    //             /** Iterate over all pixels in image (b,g,r) **/
    //             for (size_t pid = 0; pid < image_size; pid++) {
    //                 /** Iterate over all channels **/
    //                 for (size_t ch = 0; ch < num_channels; ++ch) {
    //                     /**          [images stride + channels stride + pixel id ] all in bytes            **/
    //                     data[image_id * image_size * num_channels + ch * image_size + pid] =
    //                         imagesData.at(image_id).get()[pid * num_channels + ch];
    //                 }
    //             }
    //         }
    //     }
    //     inputInfo = {};
    //     // -----------------------------------------------------------------------------------------------------

    //     // --------------------------- 7. Do inference ---------------------------------------------------------
    //     slog::info << "Start inference" << slog::endl;
    //     infer_request.Infer();
    //     // -----------------------------------------------------------------------------------------------------

    //     // --------------------------- 8. Process output -------------------------------------------------------
    //     slog::info << "Processing output blobs" << slog::endl;

    //     const Blob::Ptr outputBlob = infer_request.GetBlob(firstOutputName);

    //     /** Validating -nt value **/
    //     const size_t resultsCnt = outputBlob->size() / batchSize;
    //     if (FLAGS_nt > resultsCnt || FLAGS_nt < 1) {
    //         slog::warn << "-nt " << FLAGS_nt << " is not available for this network (-nt should be less than "
    //                    << resultsCnt + 1 << " and more than 0).\n           Maximal value " << resultsCnt << " will be used.";
    //         FLAGS_nt = resultsCnt;
    //     }

    //     /** Read labels from file (e.x. LeNet.labels) **/
    //     std::string labelFileName = fileNameNoExt(FLAGS_m) + ".labels";
    //     std::vector<std::string> labels;

    //     std::ifstream inputFile;
    //     inputFile.open(labelFileName, std::ios::in);
    //     if (inputFile.is_open()) {
    //         std::string strLine;
    //         while (std::getline(inputFile, strLine)) {
    //             trim(strLine);
    //             labels.push_back(strLine);
    //         }
    //         inputFile.close();
    //     } else {
    //         throw std::logic_error("Cannot read label file");
    //     }

    //     ClassificationResult classificationResult(outputBlob, images, batchSize, FLAGS_nt, labels);
    //     classificationResult.print();
    } catch (const std::exception& ex) {
        slog::err << ex.what() << slog::endl;
        return EXIT_FAILURE;
    }
    // slog::info << "This sample is an API example, for performance measurements, "
    //              "use the dedicated benchmark_app tool"
    //             << slog::endl;
    return 0;
}