/*
Fractal code for CS 4380 / CS 5351

Copyright (c) 2016, Texas State University. All rights reserved.

Redistribution in source or binary form, with or without modification,
is not permitted. Use in source and binary forms, with or without
modification, is only permitted for academic use in CS 4380 or CS 5351
at Texas State University.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

Author: Martin Burtscher
*/

#include <cstdlib>
#include <sys/time.h>
#include <math.h>
#include "cs43805351.h"
#include <mpi.h>

static const double Delta = 0.005491;
static const double xMid = 0.745796;
static const double yMid = 0.105089;

int main(int argc, char *argv[])
{
  int comm_sz; // NUMBER OF PROCESSES
  int my_rank;  // PROCESS RANK
  int proc_gets;
 
  MPI_Init(&argc, &argv); // new
  MPI_Comm_size(MPI_COMM_WORLD, &comm_sz); // new code
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank); // new code
  
  // check command line
  if (my_rank == 0) {
     printf("Fractal v1.5 [MPI]\n");
     printf("The total number of processes == %d\n", comm_sz);
  }

  if (argc != 3) {
     fprintf(stderr, "usage: %s frame_width num_frames\n", argv[0]);
     exit(-1);
  }
  int width = atoi(argv[1]);
   
  if (width < 10) {
     fprintf(stderr, "error: frame_width must be at least 10\n");
     exit(-1);
  }
  int frames = atoi(argv[2]);
    
  if (frames < 1) {
     fprintf(stderr, "error: num_frames must be at least 1\n");
     exit(-1);
  }
  
  if (frames % comm_sz != 0 && my_rank == 0) { // New code
     fprintf(stderr, "error: the number of frames is NOT a multiple of number of processes\n");
     exit(-1);
  } 
  proc_gets = frames/comm_sz; // NUMBER OF FRAMES EACH PROCESS GETS

  if (my_rank == 0) {
     printf("computing %d frames of %d by %d fractal\n", frames, width, width); 
  }
  const int NOT_WHOLE_MOVIE = proc_gets * width * width, // INDICATES EACH THREAD GETS THIS AMOUNT
            PROCESS_0 = 0; // READABILITY
  
  // allocate picture array
  unsigned char* fraction_pic;// WILL BE USED FOR EVERY PROCESS TO STORE THEIR 'PIECE'
  unsigned char* whole_pic;   // WILL BE USED FOR PROCESS 0 ONLY TO MPI_GATHER EACH PIECE
  if (my_rank == PROCESS_0) {
     whole_pic = new unsigned char[frames * width * width];
  }
  fraction_pic = new unsigned char[NOT_WHOLE_MOVIE];

  // start time
  struct timeval start, end;
  MPI_Barrier(MPI_COMM_WORLD);
  gettimeofday(&start, NULL);
  int end_point = proc_gets + (proc_gets * my_rank),
      my_start = my_rank == PROCESS_0 ? 0 : end_point - proc_gets; // IF/ELSE STATEMENT
  
  // compute frames
  double delta = Delta * pow(0.99, my_start);
  for (int frame = my_start; frame < end_point; frame++) {
    delta *= 0.99;
    const double xMin = xMid - delta;
    const double yMin = yMid - delta;
    const double dw = 2.0 * delta / width;
    for (int row = 0; row < width; row++) {
      const double cy = -yMin - row * dw;
      for (int col = 0; col < width; col++) {
        const double cx = -xMin - col * dw;
        double x = cx;
        double y = cy;
        int depth = 256;
        double x2, y2;
        do {
          x2 = x * x;
          y2 = y * y;
          y = 2 * x * y + cy;
          x = x2 - y2 + cx;
          depth--;
        } while ((depth > 0) && ((x2 + y2) < 5.0));
        // START AT 0 INSTEAD OF my_start
        fraction_pic[(frame-my_start) * width * width + row * width + col] = (unsigned char) depth;
      }
    }
  }
   
    
  switch(my_rank) {
   case PROCESS_0: 
      MPI_Gather(fraction_pic, NOT_WHOLE_MOVIE, MPI_UNSIGNED_CHAR, // WILL SEND PROC 0 RESULT
        whole_pic, NOT_WHOLE_MOVIE, MPI_UNSIGNED_CHAR, PROCESS_0, MPI_COMM_WORLD); //PLACE INTO WHOLE_PIC
      break;
   default:
      MPI_Gather(fraction_pic, NOT_WHOLE_MOVIE, MPI_UNSIGNED_CHAR, // WILL SEND 2 PROC 0
        NULL, NOT_WHOLE_MOVIE, MPI_UNSIGNED_CHAR, PROCESS_0, MPI_COMM_WORLD); // N/A
      break;
  }

  // end time
  gettimeofday(&end, NULL);
 
  if (my_rank == 0) {
     double runtime = end.tv_sec + end.tv_usec / 1000000.0 - start.tv_sec - start.tv_usec / 1000000.0;
     printf("compute time: %.4f s\n", runtime);
  }

  // verify result by writing frames to BMP files
  if ((width <= 400) && (frames <= 30) && (my_rank == 0)) {
    for (int frame = 0; frame < frames; frame++) {
      char name[32];
      sprintf(name, "fractal%d.bmp", frame + 1000);
      writeBMP(width, width,&whole_pic[frame * width * width], name);
    }
  }  

  if (my_rank == 0) {
    delete [] whole_pic;
    delete [] fraction_pic;
  }
 
  MPI_Finalize();
  return 0;
}
