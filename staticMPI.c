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

    /* Master allocates image buffer */
    int *image = NULL;
    if (rank == 0) {
        image = (int *)malloc(sizeof(int) * WIDTH * HEIGHT);
    }

    /* Divide rows evenly among processes (including both master and workers)*/
    
    int rows_per_process = HEIGHT / size;
    int extra_rows = HEIGHT % size;
    
    int start_row, assigned_rows;
    
    if (rank < extra_rows) {
        start_row = rank * (rows_per_process + 1);
        assigned_rows = rows_per_process + 1;
    } else {
        start_row = rank * rows_per_process + extra_rows;
        assigned_rows = rows_per_process;
    }

    /* Each process computes its assigned rows */
    int *assigned_image = (int *)malloc(sizeof(int) * assigned_rows * WIDTH);

    for (int i = 0; i < assigned_rows; i++) {
        int global_row = start_row + i; //row in the full image
        for (int j = 0; j < WIDTH; j++) {
            struct complex c;
            c.real = (j - WIDTH / 2.0) * 4.0 / WIDTH;
            c.imag = (global_row - HEIGHT / 2.0) * 4.0 / HEIGHT;
            assigned_image[i * WIDTH + j] = cal_pixel(c);
        }
    }

    /* Gather all results at master */
    int *recv_counts = NULL;
    int *displs = NULL;
    
    if (rank == 0) {
        recv_counts = (int *)malloc(size * sizeof(int));
        displs = (int *)malloc(size * sizeof(int));
        
        for (int i = 0; i < size; i++) {
            if (i < extra_rows) {
                recv_counts[i] = (rows_per_process + 1) * WIDTH;
            } else {
                recv_counts[i] = rows_per_process * WIDTH;
            }
            
            if (i == 0) {
                displs[i] = 0;
            } else {
                displs[i] = displs[i-1] + recv_counts[i-1];
            }
        }
    }

    MPI_Gatherv(assigned_image, assigned_rows * WIDTH, MPI_INT,
                image, recv_counts, displs, MPI_INT,
                0, MPI_COMM_WORLD);

    double end_time = MPI_Wtime();
    double elapsed = end_time - start_time;

    
    if (rank == 0) {
        printf("STATIC SCHEDULING:\n");
        printf("Total execution time: %.6f seconds\n", elapsed);
        save_pgm("static.pgm", image);
        free(image);
        free(recv_counts);
        free(displs);
    }

    free(assigned_image);
    MPI_Finalize();
    return 0;
}