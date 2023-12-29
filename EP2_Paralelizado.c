#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <time.h>

#define MAX_DATA_POINTS 192
#define MAX_TRAIN_POINTS 5000
#define INITIAL_POINTS  500

void merge(int *sortedIndices, float *distances, int low, int mid, int high) {
    int *temp = malloc((high - low + 1) * sizeof(int));
    int i = low, j = mid + 1, k = 0;

    while (i <= mid && j <= high) {
        if (distances[sortedIndices[i]] <= distances[sortedIndices[j]]) {
            temp[k++] = sortedIndices[i++];
        } else {
            temp[k++] = sortedIndices[j++];
        }
    }

    while (i <= mid) {
        temp[k++] = sortedIndices[i++];
    }

    while (j <= high) {
        temp[k++] = sortedIndices[j++];
    }

    for (i = low, k = 0; i <= high; i++, k++) {
        sortedIndices[i] = temp[k];
    }

    free(temp);
}


void mergeSort(int *sortedIndices, float *distances, int low, int high) {
    if (low < high) {
        int mid = (low + high) / 2;
        #pragma omp task firstprivate (sortedIndices, distances, low, mid)
        mergeSort(sortedIndices, distances, low, mid);
        #pragma omp task firstprivate (sortedIndices, distances, mid, high)
        mergeSort(sortedIndices, distances, mid + 1, high);
        #pragma omp taskwait
        merge(sortedIndices, distances, low, mid, high);
    }
}

int knn(int k, float *xtrain, float *ytrain, float *xtest, int *classCount)  {
    
    int *sortedIndices = malloc(MAX_TRAIN_POINTS*(sizeof(int)));
    float *distances = malloc(MAX_TRAIN_POINTS*(sizeof(float)));
    
    #pragma omp parallel for
    for (int i = 0; i < MAX_TRAIN_POINTS; i++) { /*Para cada linha i do arquivo xtrain*/
        float sum=0;
        for (int j = 0; j < 8; j++) {
            float delta = xtrain[i*8 + j] - xtest[j];
            float sqr = delta*delta;
            sum = sum + sqr;
        }
        distances[i] = sum;
        sortedIndices[i] = i;
    }
    // ordenação usando o mergesort
    #pragma omp parallel
    {
        #pragma omp single
        mergeSort(sortedIndices, distances, 0, MAX_TRAIN_POINTS - 1);
    }

    /*Conta as classes dos k vizinhos*/
    for (int i = 0; i < k; i++) {
        int neighborIndex = sortedIndices[i];
        int neighborClass = (int)ytrain[neighborIndex];
        #pragma omp atomic
        classCount[neighborClass]++;
    }

    free(sortedIndices);
    free(distances);

    int predictedClass = (classCount[0] > classCount[1]) ? 0 : 1;

    return predictedClass;
}

int main() {
    int k;
   
    char line[11000];
    printf("Digite o valor de k: ");
    scanf("%d", &k);

    FILE *outputFile = fopen("C:\\Users\\gabri\\Desktop\\C C++\\EP2 OAC2\\output\\output.txt", "w");

    if (outputFile == NULL) {
        printf("Erro ao abrir o arquivo de saída.\n");
        return 1;
    }

    FILE *xtrainFile = fopen("C:\\Users\\gabri\\Desktop\\C C++\\EP2 OAC2\\xtrain5000.txt", "r");
    FILE *ytrainFile = fopen("C:\\Users\\gabri\\Desktop\\C C++\\EP2 OAC2\\ytrain5000.txt", "r");
    FILE *xtestFile = fopen("C:\\Users\\gabri\\Desktop\\C C++\\EP2 OAC2\\xtest.txt", "r");

    if (xtrainFile == NULL || ytrainFile == NULL || xtestFile == NULL) {
        printf("Erro ao abrir arquivos de treinamento/teste.\n");
        return 1;
    }

    int allocatedPoints = INITIAL_POINTS;
    float *xtrain = malloc(8*allocatedPoints*sizeof(float));
    float *ytrain = malloc(8*allocatedPoints*sizeof(float));
    float xtest[8*MAX_DATA_POINTS];




    // Ler os dados de treinamento de xtrain.txt
    int numTrainPoints = 0;

    while (fgets(line, sizeof(line), xtrainFile) != NULL) {
        // Divida a linha em tokens separados por vírgula
        char *token = strtok(line, ",");

        while (token != NULL) {
            if (numTrainPoints >= allocatedPoints) {
                allocatedPoints *= 2;
                xtrain = realloc(xtrain, 8*allocatedPoints*sizeof(float));
                if (xtrain == NULL) {
                    printf("Erro ao realocar memória para xtrain.\n");
                    return 1;
                }
            }
            xtrain[numTrainPoints] = strtof(token, NULL);
            numTrainPoints++;
            token = strtok(NULL, ",");
        }
    }

    fclose(xtrainFile);

    // Reset allocatedPoints for ytrain
    allocatedPoints = INITIAL_POINTS;

   // Ler os dados de ytrain.txt
   numTrainPoints = 0;
   ytrain = malloc(allocatedPoints*sizeof(float)); // Alocar memória para ytrain com base em allocatedPoints
   while (fgets(line, sizeof(line), ytrainFile) != NULL) {
   	 // Divida a linha em tokens separados por vírgula
    	char *token = strtok(line, "\n");
    	while (token != NULL) {
           if (numTrainPoints >= allocatedPoints) {
               allocatedPoints *= 2;
               ytrain = realloc(ytrain, allocatedPoints*sizeof(float)); // Realloc memória para ytrain com base em allocatedPoints
               if (ytrain == NULL) {
                printf("Erro ao realocar memória para ytrain.\n");
                return 1;
               }
           }
           ytrain[numTrainPoints] = strtof(token, NULL);
           numTrainPoints++;
           token = strtok(NULL, ",");
        } 
     }

    fclose(ytrainFile);

    // Ler os dados de teste
    int numTestPoints = 0;

    while (fgets(line, sizeof(line), xtestFile) != NULL) {
        // Divida a linha em tokens separados por vírgula
        char *token = strtok(line, ",");

        while (token != NULL) {
            xtest[numTestPoints] = strtof(token, NULL);
            numTestPoints++;
            token = strtok(NULL, ",");
        }
    }

    fclose(xtestFile);

    // Agora o array xtrain contém os valores lidos do arquivo

    float xtest_l [8];
    int n = 0;
    int predictedClasses[MAX_DATA_POINTS];
    // Iniciar a medição do tempo
    clock_t start = clock(); 
    #pragma omp parallel for
    for (int i = 0; i < MAX_DATA_POINTS; i++) {
    int classCount[2] = {0, 0};
    float xtest_l[8];
    for (int j = 0; j < 8; j++) {
        xtest_l[j] = xtest[i*8 + j];
    }
    predictedClasses[i] = knn(k, xtrain, ytrain, xtest_l, classCount);
    }

// Escrever as classes previstas no arquivo de saída após o loop paralelo
for (int i = 0; i < MAX_DATA_POINTS; i++) {
    fprintf(outputFile, "%d.0 \n",  predictedClasses[i]);
}
    fclose(outputFile);
    // Finalizar a medição do tempo
    clock_t end = clock();

    // Calcular o tempo de execução em segundos
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Tempo de execucao: %f segundos\n", time_spent);

    // Liberar a memória alocada
    free(xtrain);
    free(ytrain);

    return 0;
}