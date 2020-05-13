#include "accelerometer_handler.h"
#include "config.h"
#include "magic_wand_model_data.h"

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

#include "mbed.h"
#include <cmath>
#include "uLCD_4DGL.h"
#include "DA7212.h"

#define bufferLength (32)
#define signalLength (42)
//int signal[signalLength];
//int16_t waveform[kAudioTxBufferSize];
char serialInBuffer[bufferLength];
int serialCount = 0;
DigitalOut led1(LED1);
DigitalOut led2(LED2);
DigitalOut led3(LED3);
DA7212 audio;
int16_t waveform[kAudioTxBufferSize];
EventQueue queue(32 * EVENTS_EVENT_SIZE);
EventQueue queue1(32 * EVENTS_EVENT_SIZE);
Thread t;
Serial pc(USBTX, USBRX);
//Thread t1;
Thread t1(osPriorityNormal, 120 * 1024 /*120K stack size*/);
InterruptIn button2(SW2);
InterruptIn button3(SW3);
uLCD_4DGL uLCD(D1, D0, D2); // serial tx, serial rx, reset pin;

int pause = 0;
int gesture_index;
int Comfirm;

// Return the result of the last prediction
int PredictGesture(float* output) {
  // How many times the most recent gesture has been matched in a row
  static int continuous_count = 0;
  // The result of the last prediction
  static int last_predict = -1;

  // Find whichever output has a probability > 0.8 (they sum to 1)
  int this_predict = -1;
  for (int i = 0; i < label_num; i++) {
    if (output[i] > 0.8) this_predict = i;
  }

  // No gesture was detected above the threshold
  if (this_predict == -1) {
    continuous_count = 0;
    last_predict = label_num;
    return label_num;
  }

  if (last_predict == this_predict) {
    continuous_count += 1;
  } else {
    continuous_count = 0;
  }
  last_predict = this_predict;

  // If we haven't yet had enough consecutive matches for this gesture,
  // report a negative result
  if (continuous_count < config.consecutiveInferenceThresholds[this_predict]) {
    return label_num;
  }
  // Otherwise, we've seen a positive result, so clear all our variables
  // and report it
  continuous_count = 0;
  last_predict = -1;

  return this_predict;
}

void DNN() {

  // Create an area of memory to use for input, output, and intermediate arrays.
  // The size of this will depend on the model you're using, and may need to be
  // determined by experimentation.
  constexpr int kTensorArenaSize = 60 * 1024;
  uint8_t tensor_arena[kTensorArenaSize];

  // Whether we should clear the buffer next time we fetch data
  bool should_clear_buffer = false;
  bool got_data = false;

  // The gesture index of the prediction

  // Set up logging.
  static tflite::MicroErrorReporter micro_error_reporter;
  tflite::ErrorReporter* error_reporter = &micro_error_reporter;

  // Map the model into a usable data structure. This doesn't involve any
  // copying or parsing, it's a very lightweight operation.
  const tflite::Model* model = tflite::GetModel(g_magic_wand_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    error_reporter->Report(
        "Model provided is schema version %d not equal "
        "to supported version %d.",
        model->version(), TFLITE_SCHEMA_VERSION);
    return -1;
  }

  // Pull in only the operation implementations we need.
  // This relies on a complete list of all the ops needed by this graph.
  // An easier approach is to just use the AllOpsResolver, but this will
  // incur some penalty in code space for op implementations that are not
  // needed by this graph.
  static tflite::MicroOpResolver<6> micro_op_resolver;
  micro_op_resolver.AddBuiltin(
      tflite::BuiltinOperator_DEPTHWISE_CONV_2D,
      tflite::ops::micro::Register_DEPTHWISE_CONV_2D());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_MAX_POOL_2D,
                               tflite::ops::micro::Register_MAX_POOL_2D());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_CONV_2D,
                               tflite::ops::micro::Register_CONV_2D());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_FULLY_CONNECTED,
                               tflite::ops::micro::Register_FULLY_CONNECTED());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_SOFTMAX,
                               tflite::ops::micro::Register_SOFTMAX());
  micro_op_resolver.AddBuiltin(tflite::BuiltinOperator_RESHAPE,
                             tflite::ops::micro::Register_RESHAPE(), 1);                             

  // Build an interpreter to run the model with
  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize, error_reporter);
  tflite::MicroInterpreter* interpreter = &static_interpreter;

  // Allocate memory from the tensor_arena for the model's tensors
  interpreter->AllocateTensors();

  // Obtain pointer to the model's input tensor
  TfLiteTensor* model_input = interpreter->input(0);
  if ((model_input->dims->size != 4) || (model_input->dims->data[0] != 1) ||
      (model_input->dims->data[1] != config.seq_length) ||
      (model_input->dims->data[2] != kChannelNumber) ||
      (model_input->type != kTfLiteFloat32)) {
    error_reporter->Report("Bad input tensor parameters in model");
    return -1;
  }

  int input_length = model_input->bytes / sizeof(float);

  TfLiteStatus setup_status = SetupAccelerometer(error_reporter);
  if (setup_status != kTfLiteOk) {
    error_reporter->Report("Set up failed\n");
    return -1;
  }

  error_reporter->Report("Set up successful...\n");

  while (true) {

    // Attempt to read new data from the accelerometer
    got_data = ReadAccelerometer(error_reporter, model_input->data.f,
                                 input_length, should_clear_buffer);

    // If there was no new data,
    // don't try to clear the buffer again and wait until next time
    if (!got_data) {
      should_clear_buffer = false;
      continue;
    }

    // Run inference, and report any error
    TfLiteStatus invoke_status = interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
      error_reporter->Report("Invoke failed on index: %d\n", begin_index);
      continue;
    }

    // Analyze the results to obtain a prediction
    gesture_index = PredictGesture(interpreter->output(0)->data.f);

    // Clear the buffer next time we read data
    should_clear_buffer = gesture_index < label_num;

    // Produce an output
    if (gesture_index < label_num) {
      error_reporter->Report(config.output_message[gesture_index]);
    }
  }
}
int song[42] = {  // song 2
  261, 392, 261, 392, 330, 440, 440,
  330, 330, 349, 349, 261, 294, 261,
  392, 392, 349, 349, 330, 330, 294,
  392, 392, 349, 349, 330, 330, 294,
  261, 261, 392, 392, 440, 440, 392,
  349, 349, 330, 330, 294, 294, 261};

int noteLength[42] = {
  1, 1, 1, 1, 1, 1, 2,
  1, 1, 1, 1, 1, 1, 2,
  1, 1, 1, 1, 1, 1, 2,
  1, 1, 1, 1, 1, 1, 2,
  1, 1, 1, 1, 1, 1, 2,
  1, 1, 1, 1, 1, 1, 2};
int song2[42] = {   // song 1
  392, 392, 349, 349, 330, 330, 294,
  392, 392, 349, 349, 330, 330, 294,
  349, 349, 294, 294, 261, 261, 247,
  261, 261, 392, 392, 440, 440, 392};
int noteLength2[42] = {
  1, 1, 1, 1, 1, 1, 2,
  1, 1, 1, 1, 1, 1, 2,
  1, 1, 1, 1, 1, 1, 2,
  1, 1, 1, 1, 1, 1, 2,
  1, 1, 1, 1, 1, 1, 2,
  1, 1, 1, 1, 1, 1, 2};
int song3[27] = {  // song 3
  261, 261, 261, 330, 392, 392 ,392, 392, 440, 440, 440, 494, 392,
  349, 349, 349, 440, 330, 330, 330, 330, 294, 294, 294, 294, 392, 392};
int noteLength3[27] = {
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2,
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2};

void printFB(){
 /* while(1){
  uLCD.locate(0,1);
  uLCD.printf("forward:");
  uLCD.locate(0,2);
  uLCD.printf("backward:");*/
  //wait(5.0);
  //}
  //led1 = !led1;
}
void select(){
  /*uLCD.locate(0,1);
  uLCD.printf("forward:");
  uLCD.locate(0,2);
  uLCD.printf("backward:");*/
  //queue1.call(printFB);
  pause = !pause;
  led1 = !led1;
}
void confirm(){
  Comfirm = 1;
  led2 = 1;
  wait(0.1);
}
void playNote(int freq)
{
  for(int i = 0; i < kAudioTxBufferSize; i++)
  {
    waveform[i] = (int16_t) (sin((double)i * 2. * M_PI/(double) (kAudioSampleFrequency / freq)) * ((1<<16) - 1));
  }
  audio.spk.play(waveform, kAudioTxBufferSize);
}
void loadSignal(void)
{
  led3 = 0;
  int i = 0;
  serialCount = 0;
  audio.spk.pause();
  while(i < signalLength)
  {
    if(pc.readable())
    {
      serialInBuffer[serialCount] = pc.getc();
      serialCount++;
      if(serialCount == 3)
      {
        //serialInBuffer[serialCount] = '\0';
        //song[i] = (int) atof(serialInBuffer);
        serialCount = 0;
        //pc.printf("%d ",song[i]);
        i++;
      }
      //pc.printf("%d ",song[i]);
    }
  }
  led3 = 1;
}
void loadSignalHandler(void) {queue.call(loadSignal);}
int gesture = 0;
int songidx = 0;
int main(void)
{
  
  //led2 = 0;
  t.start(callback(&queue, &EventQueue::dispatch_forever));
  t1.start(DNN);
  //button.rise(queue1.event(select));
  button2.rise(&select);
  button3.fall(&confirm);
  uLCD.locate(0,0);
  //uLCD.printf("song2 note:");
  uLCD.text_width(1); //4X size text
  uLCD.text_height(1);
  led1 = 1;
  led2 = 1;
  queue.call(&loadSignalHandler);
  while(led3 == 0){
    wait(0.1);
  }
  led1 = 0;
  led2 = 0;
  if(pause == 0){
  for(int i = 0; i < 42; i++)
  {
    int length = noteLength[i];
    while(length--)
    {
      uLCD.locate(0,0);
      uLCD.printf("song2 note:%2D",song[i]);
      // the loop below will play the note for the duration of 1s
      for(int j = 0; j < kAudioSampleFrequency / kAudioTxBufferSize; ++j)
      {
        queue.call(playNote, song[i]);
      }
      if(pause == 1){break;}
      if(length < 1) wait(1.0);
    }
    if(pause == 1){break;}
  }
  } 
  while(1){
      uLCD.cls();
      uLCD.locate(0,1);
      uLCD.printf("forward: song1");
      uLCD.locate(0,2);
      uLCD.printf("backward: song3:");
      uLCD.locate(0,3);
      uLCD.printf("change song");
      gesture_index = 3;
      while(pause == 1){
        queue.call(playNote, 0);
        wait(0.1);
        if(gesture_index == 0 && Comfirm == 0){gesture = 0;  uLCD.locate(0,6);  uLCD.printf("  %d",gesture);}
        if(gesture_index == 1 && Comfirm == 0){gesture = 1;  uLCD.locate(0,6);  uLCD.printf("  %d",gesture);}
        if(gesture_index == 2 && Comfirm == 0){gesture = 2;  uLCD.locate(0,6);  uLCD.printf("  %d",gesture);}
      }
      //gesture = 0;
      if(gesture == 0 && Comfirm == 1){
        uLCD.cls();
        Comfirm = 0;
        led2 = 0;
        for(int i = 0; i < 42; i++)
        {
          int length = noteLength2[i];
          while(length--)
          {
            uLCD.locate(0,0);
            uLCD.printf("song1 note: %2D",song2[i]);
            uLCD.printf("\n%d",gesture);
            // the loop below will play the note for the duration of 1s
            for(int j = 0; j < kAudioSampleFrequency / kAudioTxBufferSize; ++j)
            {
              queue.call(playNote, song2[i]);
            }
            if(pause == 1){break;}
            if(length < 1) wait(1.0);
          }
          if(pause == 1){
                queue.call(playNote, 0);
                break;}
        }

      }else if (gesture == 1 && Comfirm == 1){
        uLCD.cls();
        Comfirm = 0;
        led2 = 0;
        for(int i = 0; i < 27; i++)
        {
          int length = noteLength3[i];
          while(length--)
          {
            uLCD.locate(0,0);
            uLCD.printf("song3 note: %2D",song3[i]);
            uLCD.printf("\n%d",gesture);
            // the loop below will play the note for the duration of 1s
            for(int j = 0; j < kAudioSampleFrequency / kAudioTxBufferSize; ++j)
            {
              queue.call(playNote, song3[i]);
            }
            if(pause == 1){break;}
            if(length < 1) wait(1.0);
          }
          if(pause == 1){
              queue.call(playNote, 0);
              break;}
        }

      }else if (gesture == 2 && Comfirm == 1){
            uLCD.cls();
            Comfirm = 0;
            led2 = 0;
            uLCD.locate(0,0);
            uLCD.printf("song1");
            uLCD.printf("\nsong2");
            uLCD.printf("\nsong3");
            uLCD.printf("\n%d",gesture);
            pause = 1;
            led1 = 1;
            gesture = 3;
            while(Comfirm == 0){
              while(pause == 1){
                queue.call(playNote, 0);
                wait(0.1);
                if(gesture_index == 0 && Comfirm == 0){
                  gesture = 0;
                  songidx = 0;
                  uLCD.locate(0,6);  uLCD.printf("  %d",gesture);
                }
                if(gesture_index == 1 && Comfirm == 0){
                  gesture = 1;
                  songidx = 1;
                  uLCD.locate(0,6);  uLCD.printf("  %d",gesture);
                }
                if(gesture_index == 2 && Comfirm == 0){
                  gesture = 2;
                  songidx = 2;
                  uLCD.locate(0,6);  uLCD.printf("  %d",gesture);
                }
              }
              led2 = 0;
            }
            if (songidx == 0){
              uLCD.cls();
              for(int i = 0; i < 42; i++)
                {
                  int length = noteLength2[i];
                  while(length--)
                  {
                    uLCD.locate(0,0);
                    uLCD.printf("song1 note: %2D",song2[i]);
                    // the loop below will play the note for the duration of 1s
                    for(int j = 0; j < kAudioSampleFrequency / kAudioTxBufferSize; ++j)
                    {
                      queue.call(playNote, song2[i]);
                    }
                    if(pause == 1){break;}
                    if(length < 1) wait(1.0);
                  }
                  if(pause == 1){break;}
                }
            }
            if (songidx == 1){
                uLCD.cls();
              for(int i = 0; i < 42; i++)
                {
                  int length = noteLength[i];
                  while(length--)
                  {
                    uLCD.locate(0,0);
                    uLCD.printf("song1 note: %2D",song[i]);
                    // the loop below will play the note for the duration of 1s
                    for(int j = 0; j < kAudioSampleFrequency / kAudioTxBufferSize; ++j)
                    {
                      queue.call(playNote, song[i]);
                    }
                    if(pause == 1){break;}
                    if(length < 1) wait(1.0);
                  }
                  if(pause == 1){break;}
                }
            }
            if(songidx == 2){
              uLCD.cls();
              for(int i = 0; i < 27; i++)
                {
                  int length = noteLength3[i];
                  while(length--)
                  {
                    uLCD.locate(0,0);
                    uLCD.printf("song1 note: %2D",song3[i]);
                    // the loop below will play the note for the duration of 1s
                    for(int j = 0; j < kAudioSampleFrequency / kAudioTxBufferSize; ++j)
                    {
                      queue.call(playNote, song3[i]);
                    }
                    if(pause == 1){break;}
                    if(length < 1) wait(1.0);
                  }
                  if(pause == 1){break;}
                }
            }

      }
      else {
        wait(0.1);
      }

  }
}