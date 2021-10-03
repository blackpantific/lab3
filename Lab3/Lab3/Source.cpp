#include <stdio.h>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <omp.h>
//#include <CL/opencl.h> 
#define CL_TARGET_OPENCL_VERSION 120
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>
#include "conio.h"
#include <vector>
#include <string>

#define _CRT_SECURE_NO_WARNINGS

using namespace std;

void get_kernel_code_from_file();

void get_matrixs_from_file_v2(
	string input_file_path,
	int NKM[],
	float*& matrix1,
	float*& matrix2,
	float*& resultMatrix,
	int multiplicity);

cl_device_id InformationAboutDevice(
	cl_platform_id* platformID,
	int numberOfDevice);

void write_matrix_to_file();


int numberOfDevice = 0;//by default
string pathInputFile = "C:\\Users\\black\\Desktop\\matrix.txt";
string pathOutputFile = "C:\\Users\\black\\Desktop\\matrixResult.txt";

int NKM[3] = { 0,0,0 };
int NKMBase[3] = { 0,0,0 };//до добавления дополнительных нулей
float* matrix1 = 0;
float* matrix2 = 0;
float* resultMatrix = 0;

cl_platform_id platformID;
cl_device_id deviceID;
cl_int status;
cl_context context;
cl_command_queue queue;
cl_program program;
size_t param_value = 0;
cl_kernel kernel = NULL;

char* buf = NULL;
const char* buf_p;
size_t sizeBuf;

cl_mem arg_buffer_a;
cl_mem arg_buffer_b;
cl_mem arg_buffer_c;

int globalWorkSize = 4;//для больших матриц равно 32
int localWorkSize = 4;//для больших по 16
int threadCalculateUnits = 2;

int main()
{
	get_matrixs_from_file_v2(pathInputFile, NKM, matrix1, matrix2, resultMatrix, globalWorkSize);

	deviceID = InformationAboutDevice(&platformID, numberOfDevice);

	cl_context_properties properties[3] = { CL_CONTEXT_PLATFORM , (cl_context_properties)platformID, 0 };

	context = clCreateContext(properties, 1, &deviceID, NULL, NULL, &status);
	if (!context)
	{
		throw "Error: Failed to create a compute context!\n";
	}

	get_kernel_code_from_file();

	queue = clCreateCommandQueue(context, deviceID, CL_QUEUE_PROFILING_ENABLE, &status);
	if (!queue)
	{
		throw "Error: Failed to create a queue!\n";
	}

	program = clCreateProgramWithSource(context, 1, &buf_p, &sizeBuf, &status);
	if (!program)
	{
		throw "Error: Failed to create a program!\n";
	}

	int numOfSubmatrixes = globalWorkSize / localWorkSize;

	const string param_s = "-D LOCALWS=" + to_string(localWorkSize) + " -D NUM_OF_SUBMATRIX=" + to_string(numOfSubmatrixes);//"-D COLSROWS=2 -D PSG=2";
	int size = param_s.size();
	char* parameters = new char[size + 1];
	strcpy_s(parameters, size + 1, param_s.c_str());

	status = clBuildProgram(program, 1, &deviceID, parameters, NULL, NULL);

	status = clGetProgramBuildInfo(program, deviceID, CL_PROGRAM_BUILD_LOG, NULL, NULL, &param_value);

	char* log = NULL;
	if (param_value != 0)
	{
		log = (char*)malloc(sizeof(char));
		status = clGetProgramBuildInfo(program, deviceID, CL_PROGRAM_BUILD_LOG, param_value, log, NULL);
		printf("\n%s", log);
	}
	if (status != CL_SUCCESS)
	{
		throw "Error: Failed to build a program!\n";
	}


	kernel = clCreateKernel(program, "matrix_local_custom", &status);//ошибка обнаруживается тут
	if (!kernel || status != CL_SUCCESS)
	{
		throw "Error: Failed to create compute kernel!\n";
	}

	double start_time, end_time;

	start_time = omp_get_wtime();//отсчет времени от начала передачи данных с хоста на девайс

	/// <summary>
	/// Буффер находится на девайсе, поэтому передача данных с хоста на девайс выполняется уже
	/// в функции clEnqueueWriteBuffer
	/// </summary>
	arg_buffer_a = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * NKM[1] * NKM[2], NULL, &status);

	status = clEnqueueWriteBuffer(queue, arg_buffer_a, CL_FALSE, 0, sizeof(float) * NKM[1] * NKM[2],
		matrix1, 0, NULL, NULL);

	arg_buffer_b = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(float) * NKM[0] * NKM[1], NULL, &status);

	status = clEnqueueWriteBuffer(queue, arg_buffer_b, CL_FALSE, 0, sizeof(float) * NKM[0] * NKM[1],
		matrix2, 0, NULL, NULL);

	arg_buffer_c = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float) * NKM[0] * NKM[2], NULL, &status);

	status = clEnqueueWriteBuffer(queue, arg_buffer_c, CL_FALSE, 0, sizeof(float) * NKM[0] * NKM[2],
		resultMatrix, 0, NULL, NULL);
	if (status != CL_SUCCESS)
	{
		throw "Error: Failed in clEnqueueWriteBuffer!\n";
	}


	auto widthA = NKM[1];
	auto widthB = NKM[0];
	status = clSetKernelArg(kernel, 0, sizeof(cl_mem), &arg_buffer_a);
	status |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &arg_buffer_b);
	status |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &arg_buffer_c);
	status |= clSetKernelArg(kernel, 3, sizeof(int), &widthA);
	status |= clSetKernelArg(kernel, 4, sizeof(int), &widthB);
	status |= clSetKernelArg(kernel, 5, sizeof(int), &threadCalculateUnits);
	if (status != CL_SUCCESS)
	{
		throw "Error: Failed in clSetKernelArg!\n";
	}


	size_t dimentions = 2;
	size_t global_work_size[2];

	global_work_size[0] = NKM[2] / 2;
	global_work_size[1] = NKM[0];


	size_t local_work_size[2];

	local_work_size[0] = localWorkSize / 2;
	local_work_size[1] = localWorkSize;

	cl_event ourEvent = 0;

	status = clEnqueueNDRangeKernel(queue, kernel, dimentions, NULL, global_work_size, local_work_size, 0,
		NULL, &ourEvent);
	if (status != CL_SUCCESS)
	{
		throw "Error: Failed in clEnqueueNDRangeKernel!\n";
	}

	/// <summary>
	/// В данной функции мы получаем данные с девайса на хост, поэтому после этой функции мы заканчиваем
	/// подсчет общего времени выполнения
	/// </summary>
	status = clEnqueueReadBuffer(queue, arg_buffer_c, CL_TRUE, 0,
		sizeof(float) * NKM[0] * NKM[2], resultMatrix, 0, NULL, NULL);//самый последний ReadBuffer должен быть синхронным(CL_TRUE)
	if (status != CL_SUCCESS)
	{
		throw "Error: Failed in clEnqueueReadBuffer!\n";
	}


	for (size_t i = 0; i < NKM[0] * NKM[2]; i++)
	{
		//printf("\nc[%i] = %f", i, resultMatrix[i]);
	}


	end_time = omp_get_wtime();
	auto timeSingle = end_time - start_time;

	cl_ulong gstart, gend;

	status = clGetEventProfilingInfo(ourEvent, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &gstart, NULL);
	status = clGetEventProfilingInfo(ourEvent, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &gend, NULL);

	double nanoSeconds = gend - gstart;
	printf("\nTime: %f\t%f \n", nanoSeconds / 1000000.0, timeSingle * 1000);


	write_matrix_to_file();

	return 0;
}


void write_matrix_to_file() {

	// WRITE RESULT MATRIX TO FILE
	int numbersToRemoveOnY = NKM[2] - NKMBase[2];
	int numbersToRemoveOnX = NKM[0] - NKMBase[0];
	//printf("%d", numbersToRemoveOnY);
	//printf("%d", numbersToRemoveOnX);

	auto matrix1Rows = NKM[2];
	auto matrix1Columns = NKM[1];
	auto matrix2Rows = NKM[1];
	auto matrix2Columns = NKM[0];

	vector<char> outputData;

	string tmp = to_string(NKMBase[0]);
	char const* N = tmp.c_str();

	string tmp1 = to_string(NKMBase[2]);
	char const* M = tmp1.c_str();

	outputData.insert(outputData.end(), N, N + strlen(N));
	outputData.push_back(' ');
	outputData.insert(outputData.end(), M, M + strlen(M));
	outputData.push_back('\r');
	outputData.push_back('\n');

	int increment = 0;
	for (size_t i = 0; i < matrix1Rows; i++)//мнимые циклы
	{
		for (size_t j = 0; j < matrix2Columns; j++)
		{
			char* char_arr;
			string str_obj(to_string(resultMatrix[increment]));
			char_arr = &str_obj[0];


			outputData.insert(outputData.end(), char_arr, char_arr + strlen(char_arr));
			outputData.push_back(' ');

			increment++;
		}
		//increment += numbersToRemoveOnX;
		outputData.pop_back();
		outputData.push_back('\r');
		outputData.push_back('\n');

	}
	outputData.pop_back();
	outputData.pop_back();

	char* outputArray = &outputData[0];

	fstream bin("C:\\Users\\black\\Desktop\\matrixResult.txt", ios::out | ios::binary);
	bin.write(outputArray, sizeof(char) * outputData.size());
	bin.close();
	

}

void get_kernel_code_from_file() {

	ifstream in("Program.txt", ios::binary);
	sizeBuf = in.seekg(0, ios::end).tellg();
	if (sizeBuf == -1)
		throw "File is empty";
	in.seekg(0);
	buf = new char[sizeBuf + 1];
	in.read(buf, sizeBuf);
	buf[sizeBuf] = 0;
	in.close();
	buf_p = buf;
}

void get_matrixs_from_file_v2(string input_file_path, int NKM[], float*& matrix1, float*& matrix2, float*& resultMatrix, int multiplicity)
{
	int multiplicity1 = 4;
	float** table1FromFile = NULL;
	float** table2FromFile = NULL;

	float** table1WithAddedElements = NULL;
	float** table2WithAddedElements = NULL;

	char* bufIterator = NULL;
	char* buf = NULL;

	ifstream in(input_file_path, ios::binary);
	int size = in.seekg(0, ios::end).tellg();
	if (size == -1)
		throw "File is empty";
	in.seekg(0);
	buf = new char[size + 1];
	in.read(buf, size);
	buf[size] = 0;
	bufIterator = buf;
	in.close();
	string tempString = "";

	while (true) {


		if (*bufIterator == ' ' || *bufIterator == 13) {


			if (NKM[0] == 0) {
				NKM[0] = stoi(tempString);
				tempString = "";
			}
			else
				if (NKM[1] == 0) {
					NKM[1] = stoi(tempString);
					tempString = "";
				}
				else
					if (NKM[2] == 0) {
						NKM[2] = stoi(tempString);
						tempString = "";
					}

			if (*bufIterator == ' ') {
				bufIterator++;
			}
			else {
				bufIterator++;
				bufIterator++;
				break;
			}

		}
		else {
			tempString += *bufIterator;
			bufIterator++;
		}

	}

	NKMBase[2] = NKM[2];
	NKMBase[1] = NKM[1];
	NKMBase[0] = NKM[0];


	auto matrix1RowsFromFile = NKM[2];//M
	auto matrix1ColumnsFromFile = NKM[1];//K
	auto matrix2RowsFromFile = NKM[1];//K
	auto matrix2ColumnsFromFile = NKM[0];//N

	auto matrix1RowsWithNulls = NKM[2];//M
	auto matrix1ColumnsWithNulls = NKM[1];//K
	auto matrix2RowsWithNulls = NKM[1];//K
	auto matrix2ColumnsWithNulls = NKM[0];//N


	int matrix1RowsAddedNumber = 0;
	int matrix1ColumnsAddedNumber = 0;
	int matrix2RowsAddedNumber = 0;
	int matrix2ColumnsAddedNumber = 0;


	if ((matrix1RowsFromFile % multiplicity) != 0) {
		matrix1RowsAddedNumber = multiplicity - (matrix1RowsFromFile % multiplicity);
		matrix1RowsWithNulls += matrix1RowsAddedNumber;
	}

	if ((matrix1ColumnsFromFile % multiplicity) != 0) {
		matrix1ColumnsAddedNumber = multiplicity - (matrix1ColumnsWithNulls % multiplicity);
		matrix1ColumnsWithNulls += matrix1ColumnsAddedNumber;
	}

	if ((matrix2RowsFromFile % multiplicity) != 0) {
		matrix2RowsAddedNumber = multiplicity - (matrix2RowsWithNulls % multiplicity);
		matrix2RowsWithNulls += matrix2RowsAddedNumber;
	}

	if ((matrix2ColumnsFromFile % multiplicity) != 0) {
		matrix2ColumnsAddedNumber = multiplicity - (matrix2ColumnsWithNulls % multiplicity);
		matrix2ColumnsWithNulls += matrix2ColumnsAddedNumber;
	}


	if (matrix1ColumnsWithNulls != matrix2RowsWithNulls) {
		throw "Impossible! The collsrows don't match!";
	}










	table1FromFile = (float**)calloc(matrix1RowsFromFile, sizeof(float*));

	for (int i = 0; i < matrix1RowsFromFile; i++)
	{
		table1FromFile[i] = (float*)calloc(matrix1ColumnsFromFile, sizeof(float));//table[i] - это сам указатель на будущий массив под элементы

		int j = 0;
		while (j != matrix1ColumnsFromFile)
		{
			if ((int)*bufIterator != 32 && (int)*bufIterator != 13 && (int)*bufIterator != 10 && *bufIterator != '\0')
			{
				tempString += *bufIterator;
				bufIterator++;

			}
			else
			{
				if (tempString == "")
				{
					throw "Wrong number exception";
				}
				table1FromFile[i][j] = stod(tempString);
				j += 1;
				bufIterator++;
				tempString = "";
			}
			if (j == matrix1ColumnsFromFile)
			{
				bufIterator++;
			}

		}
	}

	/*printf("\n");
	for (size_t i = 0; i < matrix1RowsFromFile; i++)
	{
		for (size_t j = 0; j < matrix1ColumnsFromFile; j++)
		{
			printf("\nmatrix[%d][%d] = %f", i, j, table1FromFile[i][j]);
		}
	}*/



	table2FromFile = (float**)calloc(matrix2RowsFromFile, sizeof(float*));

	for (int i = 0; i < matrix2RowsFromFile; i++)
	{
		table2FromFile[i] = (float*)calloc(matrix2ColumnsFromFile, sizeof(float));//table[i] - это сам указатель на будущий массив под элементы

		int j = 0;
		while (j != matrix2ColumnsFromFile)
		{
			if ((int)*bufIterator != 32 && (int)*bufIterator != 13 && (int)*bufIterator != 10 && *bufIterator != '\0')
			{
				tempString += *bufIterator;
				bufIterator++;

			}
			else
			{
				if (tempString == "")
				{
					throw "Wrong number exception";
				}
				table2FromFile[i][j] = stod(tempString);
				j += 1;
				bufIterator++;
				tempString = "";
			}
			if (j == matrix2ColumnsFromFile)
			{
				bufIterator++;
			}

		}
	}

	/*printf("\n");
	for (size_t i = 0; i < matrix2RowsFromFile; i++)
	{
		for (size_t j = 0; j < matrix2ColumnsFromFile; j++)
		{
			printf("\nmatrix[%d][%d] = %f", i, j, table2FromFile[i][j]);
		}
	}*/



	table1WithAddedElements = (float**)calloc(matrix1RowsWithNulls, sizeof(float*));

	for (size_t i = 0; i < matrix1RowsWithNulls; i++)
	{
		table1WithAddedElements[i] = (float*)calloc(matrix1ColumnsWithNulls, sizeof(float));
	}



	table2WithAddedElements = (float**)calloc(matrix2RowsWithNulls, sizeof(float*));

	for (size_t i = 0; i < matrix2RowsWithNulls; i++)
	{
		table2WithAddedElements[i] = (float*)calloc(matrix2ColumnsWithNulls, sizeof(float));
	}




	for (size_t i = 0; i < matrix1RowsFromFile; i++)
	{
		for (size_t j = 0; j < matrix1ColumnsFromFile; j++)
		{
			table1WithAddedElements[i][j] = table1FromFile[i][j];
		}

	}

	/*printf("\n");
	for (size_t i = 0; i < matrix1RowsWithNulls; i++)
	{
		for (size_t j = 0; j < matrix1ColumnsWithNulls; j++)
		{
			printf("\nmatrix[%d][%d] = %f", i, j, table1WithAddedElements[i][j]);
		}
	}*/

	for (size_t i = 0; i < matrix2RowsFromFile; i++)
	{
		for (size_t j = 0; j < matrix2ColumnsFromFile; j++)
		{
			table2WithAddedElements[i][j] = table2FromFile[i][j];
		}

	}


	//printf("\n");
	//for (size_t i = 0; i < matrix2RowsWithNulls; i++)
	//{
	//	for (size_t j = 0; j < matrix2ColumnsWithNulls; j++)
	//	{
	//		printf("\nmatrix[%d][%d] = %f", i, j, table2WithAddedElements[i][j]);
	//	}
	//}


	auto matrix1ElementsCountWithAdded = matrix1RowsWithNulls * matrix1ColumnsWithNulls;
	auto matrix2ElementsCountWithAdded = matrix2RowsWithNulls * matrix2ColumnsWithNulls;

	matrix1 = (float*)calloc(matrix1ElementsCountWithAdded, sizeof(float));
	matrix2 = (float*)calloc(matrix2ElementsCountWithAdded, sizeof(float));

	int increment = 0;

	//просто меняем i и j местами чтобы получить разные матрицы

	for (size_t i = 0; i < matrix1ColumnsWithNulls; i++)//просто меняем i и j местами чтобы получить разные матрицы
	{
		for (size_t j = 0; j < matrix1RowsWithNulls; j++)
		{
			matrix1[increment] = table1WithAddedElements[i][j];
			increment++;
		}
	}

	/*printf("\n");
	for (size_t i = 0; i < matrix1ElementsCountWithAdded; i++)
	{
		printf("\nmatrix1[%d] = %f", i, matrix1[i]);
	}*/

	increment = 0;

	for (size_t i = 0; i < matrix2ColumnsWithNulls; i++)
	{
		for (size_t j = 0; j < matrix2RowsWithNulls; j++)
		{
			matrix2[increment] = table2WithAddedElements[i][j];
			increment++;
		}
	}

	/*printf("\n");
	for (size_t i = 0; i < matrix2ElementsCountWithAdded; i++)
	{
		printf("\nmatrix2[%d] = %f", i, matrix2[i]);
	}*/

	NKM[2] = matrix1RowsWithNulls;
	NKM[1] = matrix1ColumnsWithNulls;
	NKM[0] = matrix2ColumnsWithNulls;


	int resultMatrixCapacity = matrix1RowsWithNulls * matrix2ColumnsWithNulls;

	resultMatrix = (float*)calloc(resultMatrixCapacity, sizeof(float));

	free(buf);

	for (size_t i = 0; i < matrix1RowsFromFile; i++)
	{
		free(table1FromFile[i]);
	}
	free(table1FromFile);

	for (size_t i = 0; i < matrix2RowsFromFile; i++)
	{
		free(table2FromFile[i]);
	}
	free(table2FromFile);

	for (size_t i = 0; i < matrix1RowsWithNulls; i++)
	{
		free(table1WithAddedElements[i]);
	}
	free(table1WithAddedElements);

	for (size_t i = 0; i < matrix2RowsWithNulls; i++)
	{
		free(table2WithAddedElements[i]);
	}
	free(table2WithAddedElements);


}

cl_device_id InformationAboutDevice(cl_platform_id* platformID, int numberOfDevice)
{
	cl_uint platformCount;
	int err = clGetPlatformIDs(0, NULL, &platformCount);//gets number of available platforms
	//printf("\nNumber of platforms - %i\n", platformCount);
	cl_platform_id* platforms = (cl_platform_id*)malloc(platformCount);
	clGetPlatformIDs(platformCount, platforms, NULL);//gets platform ids


	cl_uint numberOfDevices;
	cl_device_id* devices = NULL;

	vector<cl_device_id> devicesDiscreteGPU;
	vector<cl_device_id> devicesIntegratedGPU;
	vector<cl_device_id> devicesCPU;
	vector<cl_device_id> allDevicesIDs;

	int er = CL_INVALID_PLATFORM;

	const char* attributeNames[5] = { "CPU", "GPU", "ACCELERATOR", "DEFAULT", "ALL" };
	const cl_platform_info attributeTypes[5] = {
												CL_DEVICE_TYPE_CPU,
												CL_DEVICE_TYPE_GPU,
												CL_DEVICE_TYPE_ACCELERATOR,
												CL_DEVICE_TYPE_DEFAULT,
												CL_DEVICE_TYPE_ALL };

	cl_bool res = false;
	cl_uint  numberOfUnits = 0;
	size_t paramValueRet = 0;

	for (int i = 0; i < platformCount; i++)
	{
		//поиск и сортировка GPU-устройств, поддерживающих OpenCL
		err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, 0, NULL, &numberOfDevices);

		if (numberOfDevices != 0 || err == 0) {

			devices = (cl_device_id*)malloc(numberOfDevices);
			err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_GPU, numberOfDevices, devices, NULL);
			if (err != CL_SUCCESS)
			{
				throw "Error: Failed wile getting device Id!\n";
			}

			for (size_t j = 0; j < numberOfDevices; j++)//проверка видеокарты дискретная она или интегрированная
			{
				err = clGetDeviceInfo(devices[j], CL_DEVICE_HOST_UNIFIED_MEMORY, 0, NULL, &paramValueRet);
				err = clGetDeviceInfo(devices[j], CL_DEVICE_HOST_UNIFIED_MEMORY, paramValueRet, &res, NULL);
				if (err != CL_SUCCESS)
				{
					throw "Error: Failed wile getting device info!\n";
				}
				if (res == false)
				{
					devicesDiscreteGPU.push_back(devices[j]);
				}
				else {
					devicesIntegratedGPU.push_back(devices[j]);
				}
			}
		}

		numberOfDevices = 0;

		//проверка наличия поддержки OpenCL у CPU
		err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_CPU, 0, NULL, &numberOfDevices);

		if (numberOfDevices != 0 || err == 0) {

			devices = (cl_device_id*)malloc(numberOfDevices);
			err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_CPU, numberOfDevices, devices, NULL);

			for (size_t j = 0; j < numberOfDevices; j++)
			{
				devicesCPU.push_back(devices[j]);
			}
		}
	}

	allDevicesIDs.insert(allDevicesIDs.end(), devicesDiscreteGPU.begin(), devicesDiscreteGPU.end());
	allDevicesIDs.insert(allDevicesIDs.end(), devicesIntegratedGPU.begin(), devicesIntegratedGPU.end());
	allDevicesIDs.insert(allDevicesIDs.end(), devicesCPU.begin(), devicesCPU.end());


	if (numberOfDevice > allDevicesIDs.size()) {
		auto id = allDevicesIDs[0];

		clGetDeviceInfo(id, CL_DEVICE_PLATFORM, 0, NULL, &paramValueRet);
		clGetDeviceInfo(id, CL_DEVICE_PLATFORM, paramValueRet, platformID, NULL);

		return id;
	}
	else {

		auto id = allDevicesIDs[numberOfDevice];

		clGetDeviceInfo(id, CL_DEVICE_PLATFORM, 0, NULL, &paramValueRet);
		clGetDeviceInfo(id, CL_DEVICE_PLATFORM, paramValueRet, platformID, NULL);

		return id;
	}
}

