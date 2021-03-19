from tensorflow.python.compiler.tensorrt import trt_convert as trt

input_saved_model_dir = './model/pix2pixTF'
output_saved_model_dir = './model/pix2pixTF-TRT'

conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS
conversion_params = conversion_params._replace(precision_mode="FP32")

converter = trt.TrtGraphConverterV2(
    input_saved_model_dir=input_saved_model_dir,  conversion_params=conversion_params)
converter.convert()
converter.save(output_saved_model_dir)
