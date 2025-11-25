#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define WIDTH 640
#define HEIGHT 480
#define MAX_ITER 255

struct complex {
    double real;
    double imag;
};

int cal_pixel(struct complex c) {
    

            double z_real = 0;
            double z_imag = 0;

            double z_real2, z_imag2, lengthsq;

            int iter = 0;
            do {
                z_real2 = z_real * z_real;
                z_imag2 = z_imag * z_imag;

                z_imag = 2 * z_real * z_imag + c.imag;
                z_real = z_real2 - z_imag2 + c.real;
                lengthsq =  z_real2 + z_imag2;
                iter++;
            }
            while ((iter < MAX_ITER) && (lengthsq < 4.0));

            return iter;

}

void save_pgm(const char *filename, int *image) {
    FILE* pgmimg; 
    int temp;
    pgmimg = fopen(filename, "wb"); 
    fprintf(pgmimg, "P2\n"); // Writing Magic Number to the File   
    fprintf(pgmimg, "%d %d\n", WIDTH, HEIGHT);  // Writing Width and Height
    fprintf(pgmimg, "255\n");  // Writing the maximum gray value 
    int count = 0; 
    
    for (int i = 0; i < HEIGHT; i++) { 
        for (int j = 0; j < WIDTH; j++) { 
            temp = image[i * WIDTH + j]; //i*WIDTH since it is 1D array 
            fprintf(pgmimg, "%d ", temp); // Writing the gray values in the 2D array to the file 
        } 
        fprintf(pgmimg, "\n"); 
    } 
    fclose(pgmimg); 
    printf("Saved %s\n", filename);
}

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double start_time = MPI_Wtime();

    if (rank == 0) {
        /* MASTER PROCESS */
        int *image = (int *)malloc(sizeof(int) * WIDTH * HEIGHT);
        
        int next_row = 0;
        int completed_rows = 0;
        MPI_Status status;

        /* 1: Send initial work to all workers */
        for (int worker = 1; worker < size; worker++) {
            if (next_row < HEIGHT) {
                MPI_Send(&next_row, 1, MPI_INT, worker, 0, MPI_COMM_WORLD);
                next_row++;
            } else {
                int stop_signal = -1;
                MPI_Send(&stop_signal, 1, MPI_INT, worker, 0, MPI_COMM_WORLD);
            }
        }

        /* 2: Receive results and assign new work */
        while (completed_rows < HEIGHT) {
            int received_row;
            int *row_data = (int *)malloc(WIDTH * sizeof(int));
            
            /* Receive row index from worker */
            MPI_Recv(&received_row, 1, MPI_INT, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &status);
            int worker_rank = status.MPI_SOURCE;
            
            /* Receive row data from same worker */
            MPI_Recv(row_data, WIDTH, MPI_INT, worker_rank, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            
            /* Store the received row in final image */
            memcpy(&image[received_row * WIDTH], row_data, WIDTH * sizeof(int));
            free(row_data);
            
            completed_rows++;
            
            /* Send new work to this worker if available */
            if (next_row < HEIGHT) {
                MPI_Send(&next_row, 1, MPI_INT, worker_rank, 0, MPI_COMM_WORLD);
                next_row++;
            } else {
                int stop_signal = -1;
                MPI_Send(&stop_signal, 1, MPI_INT, worker_rank, 0, MPI_COMM_WORLD);
            }
            
        }

        double end_time = MPI_Wtime();
        double elapsed = end_time - start_time;

        printf("DYNAMIC SCHEDULING:\n");
        printf("Total execution time: %.6f seconds\n", elapsed);
        save_pgm("dynamic.pgm", image);
        free(image);

    } else {
        /* WORKER PROCESS */
        int assigned_row;
        MPI_Status status;
        
        while (1) {
            /* Receive work assignment from master */
            MPI_Recv(&assigned_row, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
            
            /* Check for termination signal */
            if (assigned_row == -1) {
                break;
            }
            
            /* Compute the assigned row */
            int *row_data = (int *)malloc(WIDTH * sizeof(int));
            for (int j = 0; j < WIDTH; j++) {
                struct complex c;
                c.real = (j - WIDTH / 2.0) * 4.0 / WIDTH;
                c.imag = (assigned_row - HEIGHT / 2.0) * 4.0 / HEIGHT;
                row_data[j] = cal_pixel(c);
            }
            
            /* Send back results to master */
            MPI_Send(&assigned_row, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
            MPI_Send(row_data, WIDTH, MPI_INT, 0, 2, MPI_COMM_WORLD);
            
            free(row_data);
        }
    }

    MPI_Finalize();
    return 0;
}