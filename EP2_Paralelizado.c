#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <time.h>

#define MAX_DATA_POINTS 192

//#define MAX_TRAIN_POINTS 100
//#define MAX_TRAIN_POINTS 500
//#define MAX_TRAIN_POINTS 1000
//#define MAX_TRAIN_POINTS 5000
//#define MAX_TRAIN_POINTS 10000
//#define MAX_TRAIN_POINTS 50000
//#define MAX_TRAIN_POINTS 100000
//#define MAX_TRAIN_POINTS 200000
//#define MAX_TRAIN_POINTS 500000
#define MAX_TRAIN_POINTS 1000000

#define INITIAL_POINTS  50000

/*Função que ordena e une as partes*/
void merge(int *sortedIndices, float *distances, int low, int mid, int high) {
    int *temp = malloc((high - low + 1) * sizeof(int));
    int i = low, j = mid + 1, k = 0;

    while (i <= mid && j <= high) { //Ordenação
        if (distances[sortedIndices[i]] <= distances[sortedIndices[j]]) {
            temp[k++] = sortedIndices[i++];
        } else {
            temp[k++] = sortedIndices[j++];
        }
    }

    while (i <= mid) { //Avança de posição (vetor com a primeira metade)
        temp[k++] = sortedIndices[i++];
    }

    while (j <= high) { //Avança de posição (vetor com a segunda metade)
        temp[k++] = sortedIndices[j++];
    }

    for (i = low, k = 0; i <= high; i++, k++) { //Posiciona corretamente no vetor
        sortedIndices[i] = temp[k];
    }

    free(temp);
}

/*Função que implementa a forma de ordenação (mergesort), dividindo o vetor em partes menores a serem ordenadas*/
void mergeSort(int *sortedIndices, float *distances, int low, int high) {
    if (low < high) {
        int mid = (low + high) / 2;
        #pragma omp task firstprivate (sortedIndices, distances, low, mid) //Cria uma thread para a primeira metade
        mergeSort(sortedIndices, distances, low, mid); //Manda ordenar a primeira metade
        #pragma omp task firstprivate (sortedIndices, distances, mid, high) //Cria uma thread para a segunda metade
        mergeSort(sortedIndices, distances, mid + 1, high); //Manda ordenar a segunda metade
        #pragma omp taskwait //Espera as duas metades serem ordenadas
        merge(sortedIndices, distances, low, mid, high); //para unir as partes
    }
}

/*Função knn para analisar os k vizinhos mais próximos do ponto xtest*/
int knn(int k, float *xtrain, float *ytrain, float *xtest, int *classCount)  {
    
   
    //!!!!!!!!!!!!!!!!!Definir quantas threads serão usadas (opcional)!!!!!!!!
    //omp_set_num_threads(4);
    //omp_set_num_threads(8);
    //omp_set_num_threads(12); 
    


    int *sortedIndices = malloc(MAX_TRAIN_POINTS*(sizeof(int)));
    float *distances = malloc(MAX_TRAIN_POINTS*(sizeof(float)));
    
    #pragma omp parallel for
    for (int i = 0; i < MAX_TRAIN_POINTS; i++) { /*Para cada linha i do arquivo xtrain*/
        float sum=0;
        for (int j = 0; j < 8; j++) {
            float delta = xtrain[i*8 + j] - xtest[j]; //Calcula a distância euclidiana
            float sqr = delta*delta;
            sum = sum + sqr;
        }
        distances[i] = sum; //Vetor de distâncias
        sortedIndices[i] = i; //Vetor de índices ordenados
    }
    // ordenação usando o mergesort
    #pragma omp parallel //Cria threads para a ordenação, já que cada parte é ordenada de forma independente para serem unidas depois
    {
        #pragma omp single
        mergeSort(sortedIndices, distances, 0, MAX_TRAIN_POINTS - 1);
    }

    /*Conta as classes dos k vizinhos*/
    for (int i = 0; i < k; i++) {
        int neighborIndex = sortedIndices[i];
        int neighborClass = (int)ytrain[neighborIndex];
        #pragma omp atomic //Garante que a variável compartilhada classCount seja atualizada de forma atômica
        classCount[neighborClass]++;
    }

    free(sortedIndices);
    free(distances);
    // Retorna a classe com maior contagem
    int predictedClass = (classCount[0] > classCount[1]) ? 0 : 1;

    return predictedClass;
}

int main() {
    int k;
   
    char line[11000];
    printf("Digite o valor de k: ");
    scanf("%d", &k);

    //!!!!!!!!!!!!!!!!!!!!!!!!!Corrigir para PATH do destino
    //FILE *outputFile = fopen("ytestP100.txt", "w");
    //FILE *outputFile = fopen("ytestP500.txt", "w");
    //FILE *outputFile = fopen("ytestP1000.txt", "w");
    //FILE *outputFile = fopen("ytestP5000.txt", "w");
    //FILE *outputFile = fopen("ytestP10000.txt", "w");
    //FILE *outputFile = fopen("ytestP50000.txt", "w");
    //FILE *outputFile = fopen("ytestP100000.txt", "w");
    //FILE *outputFile = fopen("ytestP200000.txt", "w");
    //FILE *outputFile = fopen("ytestP500000.txt", "w");
    FILE *outputFile = fopen("ytestP1000000.txt", "w");

    if (outputFile == NULL) {
        printf("Erro ao abrir o arquivo de saída.\n");
        return 1;
    }

    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //!!!!!!!!!!!!!!!!!!!!!!!!!Corrigir para PATH dos arquivos corretos!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    //FILE *xtrainFile = fopen("xtrain100.txt", "r");
    //FILE *xtrainFile = fopen("xtrain500.txt", "r");
    //FILE *xtrainFile = fopen("xtrain1000.txt", "r");
    //FILE *xtrainFile = fopen("xtrain5000.txt", "r");
    //FILE *xtrainFile = fopen("xtrain10000.txt", "r");
    //FILE *xtrainFile = fopen("xtrain50000.txt", "r");
    //FILE *xtrainFile = fopen("xtrain100000.txt", "r");
    //FILE *xtrainFile = fopen("xtrain200000.txt", "r");
    //FILE *xtrainFile = fopen("xtrain500000.txt", "r");
    FILE *xtrainFile = fopen("xtrain1000000.txt", "r");

    //FILE *ytrainFile = fopen("ytrain100.txt", "r");
    //FILE *ytrainFile = fopen("ytrain500.txt", "r");
    //FILE *ytrainFile = fopen("ytrain1000.txt", "r");
    //FILE *ytrainFile = fopen("ytrain5000.txt", "r");
    //FILE *ytrainFile = fopen("ytrain10000.txt", "r");
    //FILE *ytrainFile = fopen("ytrain50000.txt", "r");
    //FILE *ytrainFile = fopen("ytrain100000.txt", "r");
    //FILE *ytrainFile = fopen("ytrain200000.txt", "r");
    //FILE *ytrainFile = fopen("ytrain500000.txt", "r");
    FILE *ytrainFile = fopen("ytrain1000000.txt", "r");

    FILE *xtestFile = fopen("xtest.txt", "r");

    if (xtrainFile == NULL || ytrainFile == NULL || xtestFile == NULL) {
        printf("Erro ao abrir arquivos de treinamento/teste.\n");
        return 1;
    }

    int allocatedPoints = INITIAL_POINTS; // Quantidade de pontos alocados inicialmente (evitar sobrecarga de memória)
    float *xtrain = malloc(8*allocatedPoints*sizeof(float));
    float *ytrain = malloc(8*allocatedPoints*sizeof(float));
    float xtest[8*MAX_DATA_POINTS];




    // Ler os dados de treinamento de xtrain.txt
    int numTrainPoints = 0;

    while (fgets(line, sizeof(line), xtrainFile) != NULL) {
        // Divida a linha em tokens separados por vírgula
        char *token = strtok(line, ",");

        while (token != NULL) {
            if (numTrainPoints >= allocatedPoints) {  // Se o número de pontos de treinamento exceder allocatedPoints,
                allocatedPoints *= 2;
                xtrain = realloc(xtrain, 8*allocatedPoints*sizeof(float)); // realoque memória
                if (xtrain == NULL) {
                    printf("Erro ao realocar memória para xtrain.\n");
                    return 1;
                }
            }
            xtrain[numTrainPoints] = strtof(token, NULL); //Converte de string para float
            numTrainPoints++;
            token = strtok(NULL, ",");
        }
    }

    fclose(xtrainFile);

    // Reseta allocatedPoints para o valor inicial
    allocatedPoints = INITIAL_POINTS;

   // Ler os dados de ytrain.txt
   numTrainPoints = 0;
   ytrain = malloc(allocatedPoints*sizeof(float)); // Alocar memória para ytrain com base em allocatedPoints
   while (fgets(line, sizeof(line), ytrainFile) != NULL) {
   	 // Divida a linha em tokens separados por vírgula
    	char *token = strtok(line, "\n");
    	while (token != NULL) {
           if (numTrainPoints >= allocatedPoints) { //Quantidade inicial de pontos insuficiente
               allocatedPoints *= 2;
               ytrain = realloc(ytrain, allocatedPoints*sizeof(float)); // Realoca memória
               if (ytrain == NULL) {
                    printf("Erro ao realocar memória para ytrain.\n");
                    return 1;
                }
            }
           ytrain[numTrainPoints] = strtof(token, NULL); //Converte de string para float
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
            xtest[numTestPoints] = strtof(token, NULL); //Converte de string para float
            numTestPoints++;
            token = strtok(NULL, ",");
        }
    }
    fclose(xtestFile);

    int n = 0;
    int predictedClasses[MAX_DATA_POINTS];

    // Iniciar a medição do tempo
    clock_t start = clock(); 
    #pragma omp parallel for //Cria threads para cada chamada a knn (cada ponto calculado)
    for (int i = 0; i < MAX_DATA_POINTS; i++) {
        int classCount[2] = {0, 0};
        float xtest_l[8];
        for (int j = 0; j < 8; j++) {
            xtest_l[j] = xtest[i*8 + j]; //Avança xtest de linha em linha (de 8 em 8 valores) para fazer o knn
        }
        predictedClasses[i] = knn(k, xtrain, ytrain, xtest_l, classCount);
    }
    // Finalizar a medição do tempo
    clock_t end = clock();
    
    // Escrever as classes previstas no arquivo de saída após o loop paralelo
    for (int i = 0; i < MAX_DATA_POINTS; i++) {
        fprintf(outputFile, "%d.0 \n",  predictedClasses[i]);
    }
    fclose(outputFile);


    // Calcular o tempo de execução em segundos
    double time_spent = (double)(end - start) / CLOCKS_PER_SEC;
    printf("Tempo de execucao: %f segundos\n", time_spent);

    // Liberar a memória alocada
    free(xtrain);
    free(ytrain);

    return 0;
}
