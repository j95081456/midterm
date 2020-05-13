#include "mbed.h"
#include <cmath>
#include "uLCD_4DGL.h"
#include "DA7212.h"

DigitalOut led1(LED1);
DA7212 audio;
int16_t waveform[kAudioTxBufferSize];
EventQueue queue(32 * EVENTS_EVENT_SIZE);
EventQueue queue1(32 * EVENTS_EVENT_SIZE);
Thread t;
Thread t1;
InterruptIn button(SW2);

uLCD_4DGL uLCD(D1, D0, D2); // serial tx, serial rx, reset pin;

int pause = 0;

int song[42] = {
  261, 261, 392, 392, 440, 440, 392,
  349, 349, 330, 330, 294, 294, 261,
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
int song2[42] = {
  261, 261, 392, 392, 440, 440, 392,
  349, 349, 294, 294, 261, 261, 247,
  392, 392, 349, 349, 330, 330, 294,
  392, 392, 349, 349, 330, 330, 294};
int noteLength2[42] = {
  1, 1, 1, 1, 1, 1, 2,
  1, 1, 1, 1, 1, 1, 2,
  1, 1, 1, 1, 1, 1, 2,
  1, 1, 1, 1, 1, 1, 2,
  1, 1, 1, 1, 1, 1, 2,
  1, 1, 1, 1, 1, 1, 2};
int song3[27] = {
  261, 261, 261, 330, 392, 392 ,392, 392, 440, 440, 440, 494, 392,
  349, 349, 349, 440, 330, 330, 330, 330, 294, 294, 294, 294, 392, 392};
int noteLength3[27] = {
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2,
  1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2};

void printFB(){
  while(1){
  uLCD.locate(0,1);
  uLCD.printf("forward:");
  uLCD.locate(0,2);
  uLCD.printf("backward:");
  //wait(5.0);
  }
  //led1 = !led1;
}
void select(){
  /*uLCD.locate(0,1);
  uLCD.printf("forward:");
  uLCD.locate(0,2);
  uLCD.printf("backward:");*/
  //queue1.call(printFB);
  pause = 1;
  led1 = !led1;
}

void playNote(int freq)
{
  for(int i = 0; i < kAudioTxBufferSize; i++)
  {
    waveform[i] = (int16_t) (sin((double)i * 2. * M_PI/(double) (kAudioSampleFrequency / freq)) * ((1<<16) - 1));
  }
  audio.spk.play(waveform, kAudioTxBufferSize);
}

int main(void)
{
  t.start(callback(&queue, &EventQueue::dispatch_forever));
  t1.start(callback(&queue1, &EventQueue::dispatch_forever));
  //button.rise(queue1.event(select));
  button.rise(&select);
  uLCD.locate(0,0);
  uLCD.printf("note:");
  uLCD.text_width(1); //4X size text
  uLCD.text_height(1);
  /*uLCD.locate(0,1);
  uLCD.printf("forward:");
  uLCD.locate(0,2);
  uLCD.printf("backward:");*/
  if(pause == 0){
  for(int i = 0; i < 42; i++)
  {
    int length = noteLength[i];
    while(length--)
    {
      uLCD.locate(8,0);
      uLCD.printf("%2D",song[i]);
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
      uLCD.cls();
      uLCD.locate(0,1);
      uLCD.printf("forward:");
      uLCD.locate(0,2);
      uLCD.printf("backward:");
      uLCD.locate(0,3);
      uLCD.printf("change song:");
     
}