#include <stdlib.h>
#include <stdio.h>
#include <cmath>
#include <hip/hip_runtime.h>
#include <iostream>
#include <fstream> // To use ifstream
#include <vector>
using namespace std;
const int NumOfBins = 1024;
const int bufferSize = 1024;

vector< double> readFile() {
    std::vector< double> numbers;
    ifstream inputFile("data.txt");        // Input file stream object

    // Check if exists and then open the file.
    if (inputFile.good()) {
        // Push items into a vector
        double current_number = 0;
        while (inputFile >> current_number) {
            numbers.push_back(current_number);
        }

        // Close the file.
        inputFile.close();
    }
    else {
        cout << "Error!";
    }
    return numbers;
}

void showData(vector<double> numbers) {
    // Display the numbers read:
    cout << "The numbers are: ";
    for (int count = 0; count < numbers.size(); count++) {
        cout.precision(17);
        cout << numbers[count] << " ";
    }
}


__global__ void MaxReduceKernel(double* dOut, const double* dIn) {
    extern __shared__ double sData[];

    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    int tId = threadIdx.x;

    sData[tId] = dIn[myId];
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tId < s) {
            double a = sData[tId], b = sData[tId + s];
            sData[tId] = a > b ? a : b;
        }
        __syncthreads();
    }

    if (tId == 0) {
        dOut[blockIdx.x] = sData[0];
    }
}

__global__ void MinReduceKernel(double* dOut, const double* dIn) {
    extern __shared__ double sData[];

    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    int tId = threadIdx.x;

    sData[tId] = dIn[myId];
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tId < s) {
            double a = sData[tId], b = sData[tId + s];
            sData[tId] = a < b ? a : b;
        }
        __syncthreads();
    }

    if (tId == 0) {
        dOut[blockIdx.x] = sData[0];
    }
}


__global__ void HistoGenerator(unsigned int* binOut, const double* dIn, const double maxVal, double minVal, int size) {
    int tid = threadIdx.x + blockDim.x * blockIdx.x;

    if (tid < size) {
        int binTarget = NumOfBins * (dIn[tid] - minVal) / (maxVal - minVal);
        atomicAdd(&(binOut[binTarget]), 1);
    }
}


__global__ void histoReduce(unsigned int* reduced, const unsigned int* original){
    //performing reduction 
    extern __shared__ unsigned int RdcData[];

    int index = threadIdx.x + blockDim.x * blockIdx.x;
    int Myidx = threadIdx.x;
    RdcData[Myidx] = original[index];
    __syncthreads(); 

    for (int s = blockDim.x / 2; s > 0; s >>= 1){
        if (Myidx < s){
            RdcData[Myidx] = RdcData[Myidx]+RdcData[Myidx + s];
        }
        __syncthreads();}
    
    if (Myidx == 0){
        reduced[blockIdx.x] = RdcData[0];}
}



__global__ void pdfGenerator(const unsigned int* in, float* out, const int size, const unsigned int count){
    int iDx = threadIdx.x + blockIdx.x * blockDim.x;

    if (iDx < size){
        out[iDx] = static_cast<float>(in[iDx]) / count;
    }
}


template <class CDFScan>
__global__ void ScanPDF(CDFScan *pPdf, CDFScan *pCdf, int bufferSize){
	__shared__ CDFScan share[2*1024];
	int myElm = threadIdx.x;
	int idxL  = 0, idxR = 1 ; 
	share[myElm] = pPdf[myElm];

	__syncthreads();

    //Moving values
	for( int step = 1; step < bufferSize; step <<= 1 ){
	    idxL  = 1 - idxL ;
        idxR = 1 - idxL ;
         
        if(myElm < step){
            share[idxL *bufferSize + myElm] = share[idxR*bufferSize + myElm];
        }
		else{
			share[idxL *bufferSize + myElm] = share[idxR*bufferSize + myElm] + share[idxR*bufferSize + myElm - step];
        }
      
	__syncthreads();
	}
    //Moving new data to ouputs
	pPdf[myElm] = share[idxL *bufferSize+myElm]; 
	pCdf[myElm] = pPdf[myElm];
}




int main()
{
    //Loading data
    vector< double> data;
    data = readFile();
    //Fiding min and max
    double max;
    double min;
    //showData(data);   A function to check if the data is loaded correctly

    //Initialization
    double* dData;
    double* dReduc;
    size_t srcSize = data.size() * sizeof(double);
    size_t reduc = data.size() * sizeof(double);   

    hipMalloc(&dData, srcSize);
    hipMalloc(&dReduc, reduc);
    hipMemcpy(dData, data.data(), sizeof(double) * data.size(), hipMemcpyHostToDevice);

    //Kernel Parameters
    dim3 blockDim(1024, 1, 1);
    dim3 gridDim((data.size() / blockDim.x));
    size_t size = blockDim.x * sizeof(double);

    //Fiding Max Value
    MaxReduceKernel << <gridDim, blockDim, size >> > (dReduc, dData);
    MaxReduceKernel << <1, blockDim, size >> > (dReduc, dReduc);    
    hipMemcpy(&max, dReduc, sizeof(double), hipMemcpyDeviceToHost);

    //Finding Min value
    MinReduceKernel << <gridDim, blockDim, size >> > (dReduc, dData);
    MinReduceKernel << <1, blockDim, size >> > (dReduc, dReduc);
    hipMemcpy(&min, dReduc, sizeof(double), hipMemcpyDeviceToHost);
   
    cout.precision(20);
    cout << "Max value is : " << max << endl;
    cout << "Min value is : " << min << endl;


    //Produce the histogram 
    unsigned int* dBins;
    vector<unsigned int> bins(NumOfBins, 0);

    hipMalloc(&dBins, sizeof(unsigned int) * bins.size());
    hipMemcpy(dBins, bins.data(), sizeof(unsigned int) * bins.size(), hipMemcpyHostToDevice);
    HistoGenerator << <gridDim, blockDim >> > (dBins, dData, max, min, data.size());
    hipMemcpy(bins.data(), dBins, sizeof(unsigned int) * bins.size(), hipMemcpyDeviceToHost);


    //Sum Reduce histogram and creating PDF
    unsigned int Sum;
    unsigned int* dSum;
   
    hipMalloc(&dSum, bins.size() * sizeof(unsigned int));
    hipMemcpy(dBins, bins.data(), bins.size() * sizeof(unsigned int), hipMemcpyHostToDevice);
    histoReduce << <1, blockDim, NumOfBins * sizeof(unsigned int) >> > (dSum, dBins);
    hipMemcpy(&Sum, dSum, sizeof(unsigned int), hipMemcpyDeviceToHost);

    vector<float> pdfVals(bins.size() , 0);
    int histogram;
    float* dHistogram;

    hipMalloc(&dHistogram, bins.size() * sizeof(float));
    hipMemcpy(&histogram, dBins, sizeof(int), hipMemcpyDeviceToHost);
    dim3 HistoBlock(32);
    dim3 HistoGrid((NumOfBins + HistoBlock.x - 1) / HistoBlock.x);
    pdfGenerator << <HistoGrid, HistoBlock >> > (dBins, dHistogram,  bins.size(), Sum);   
    hipMemcpy(pdfVals.data(), dHistogram, pdfVals.size() * sizeof(float), hipMemcpyDeviceToHost);


    //generate a CDF
    ofstream myFile;
    int cdfMemSize = NumOfBins * sizeof(float);
    vector<float> cdfVals(NumOfBins);
    double binWidth = (max - min) / NumOfBins;
    float *scnPDF;
    
    hipMalloc(&scnPDF, cdfVals.size() * sizeof(float));  
 	hipMemcpy(dHistogram, pdfVals.data(), cdfVals.size() * sizeof(float),hipMemcpyHostToDevice);
    ScanPDF<<<1, NumOfBins, cdfMemSize >>>(dHistogram, scnPDF,bufferSize); 
    hipMemcpy(cdfVals.data(), scnPDF, cdfVals.size() * sizeof(float), hipMemcpyDeviceToHost);

    myFile.open("CDFdata.dat");
    myFile << "bin Vals\tPDF Vlas\tCDF Vlas\n";
    for (int a = 0; a < cdfVals.size(); a++)    {
        myFile << binWidth * a + min  << "\t" << pdfVals[a] << "\t" << cdfVals[a] << "\n";
    }

    myFile <<endl;
    myFile.close();
    cout << "'CDFdata.dat' is generated successfully" <<endl;

    

    return 0;
}
