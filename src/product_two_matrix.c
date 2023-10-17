#include <stdio.h>
#include <stdlub.h>

int n_rows = 3;
int shared_n_cols = 2;
int n_cols = 6;

//matrix_a (3, 2)
int matrix_a[6] = {1, 2, 3, 4, 5, 6};

//matrix_b (2, 6)
int matrix_b[12] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
int * matrix_c = (int*)calloc(3*6, sizeof(int));



for (int n = 0; n < n_rows; ++n) {
    for (int m = 0; m < n_cols; ++m) {
        index = n*n_cols + m;
        for (int i = 0; i < shared_n_cols; ++i) {
            int * num_a = &matrix_a[i + n*shared_n_cols];
            int * num_b = &matrix_b[m + i*n_cols];
            matrix_c[index] += *num_a * *num_b; 

            // printf("%d %d %d\n", index, *num_a, *num_b);
        }
    }
}

printf("\n\n");
for (int i = 0; i < 18; ++i) {
    printf("%d ", matrix_c[i]);
}
printf("\n\n");

// 3 rows, 6 cols
for (int n = 0; n < 3; ++n) {
    for (int m = 0; m < 6; ++m) {
        printf("%d ", matrix_c[n*6 + m]);
    }
    printf("\n");
}

free(matrix_c);
