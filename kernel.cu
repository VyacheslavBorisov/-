#define _USE_MATH_DEFINES

#define BLOCK_SIZE   16

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <cstring>
#include <math.h>
#include <ctime>
//#include <omp.h>
#include <string>

struct Kathodparam {
	float R;				//Радиус основания в 0.хх (100 - высота области)
	float r;				//Радиус вершины катода
	float step;				//Шаг по координате
	float eps;				//Высота вакуума над катодом
	int size1;				//Ширина области в точках
	int size2;				//Высота области
	int mid1;				//Половина катода по Х + 1 точка
	int mid2;				//Половина катода до оси симметрии
	int siv;				//Шагов в вакууме
	int sir;				//Шагов в малом радиусе
	int siR;				//Шагов в большем радиусе
	int sik;				
	int indent;				//Отступ от края области до нижнего основания
};


__global__ void basefunc(float* T, float* T0, float* ar, float* xcent, float* E1, int* L, float YT, float koeflaplas, float s2, float tauT, float t0, float l, float rho, float c, float r0, struct Kathodparam K);
__global__ void borderfunc1(float* T, float* T0, float* E2, float* Table, int* borderfirst, int borderfirstsize, float t0, float l, float c, float r0, float lambda, struct Kathodparam K);
__global__ void borderfunc2(float* T, float* T0, float* E2, float* Table, int* bordersecond, int bordersecondsize, float l, float cos1, float sin1, float c, float r0, float lambda, struct Kathodparam K);
__host__ void mesh(float** inout, struct Kathodparam K, float s1, float s2);
__host__ void solveall(float** inout, float* Table, float* T, float* u, float border, float* xcent, float* x, float* y, struct Kathodparam K, float s1, float s2, std::string DirName);
__host__ void derivative(float* u, int size1, int size2, float step, std::string DirName);
__device__ float jem(float* Table, float r0, float T, float E);
__device__ float Ef(float* E1, float* E2, int mid1, int size2, int sir, int siv, int i, int j);
__device__ float E(float T);
__host__ float Dcoef(float* Table, float* u, float* T, int* L, int size1, int step, int s2, float r0);
__host__ void Tablereader(float* Table);
__host__ std::string get_dir_name();
__host__ std::string get_name(std::string name, int k);
__host__ std::string get_name(std::string DirName, std::string name);
__host__ std::string get_name(std::string DirName, std::string name, int k);
__host__ void output_vtk_binary(float* u, int n1, int n2, float* x, float* y, std::string name);


int main() {
	setlocale(LC_ALL, "Russian");
	//omp_set_num_threads(8);
	float* Table;
	float** inout, * x, * y, * xcent, border[2][3], start, tau, * u, * T;
	float s1, s2, c, l;
	struct Kathodparam Kathod;
	l = 164000;
	c = 678;
	s1 = 0.001;		//Коэффициент вне катода[]
	s2 = 1000;		//Внутри катода
	std::string DirName = get_dir_name();
	system(("mkdir " + DirName).c_str());
	Kathod.R = 30;
	Kathod.r = 3;
	Kathod.eps = 3;
	border[0][0] = 1;
	border[1][0] = 4;
	border[0][1] = 1;
	border[1][1] = 0;
	border[0][2] = 2;
	border[1][2] = 0;
	start = 0;
	Kathod.sir = 5;							//Шагов в радиусе.
	Kathod.indent = 3;
	Kathod.step = Kathod.r / (Kathod.sir * 100);				//Длина шага.
	Kathod.siR = floor(Kathod.R * Kathod.sir / Kathod.r);			//Шагов в большем радиусе.
	Kathod.size1 = 2*Kathod.indent + 2 * Kathod.siR + 1;				//Размер области по горизонтали в точках, нечетное число.
	Kathod.size2 = floor(1 / Kathod.step + 1);		//Размер области по вертикали вточках.
	Kathod.siv = floor(Kathod.eps / (100 * Kathod.step));	//Число шагов в вакууме.
	Kathod.sik = 2 * Kathod.siR;						//Диаметр большего основания в шагах.
	Kathod.mid1 = (Kathod.size1 + 1) / 2;								//Центр по Х
	Kathod.mid2 = (Kathod.size1 - 1) / 2;								//Центр по Х уменьшенный(?)
	x = new float[Kathod.size1];				//Массив координат по Х.
	y = new float[Kathod.size2];				//Массив координат по У.
	for (int i = 0; i < Kathod.size1; i++)
		x[i] = i * Kathod.step;
	for (int i = 0; i < Kathod.size2; i++)
		y[i] = i * Kathod.step;
	xcent = new float[Kathod.size1];				//Массив координат по Х.
	for (int i = 0; i < Kathod.size1; i++)
		xcent[i] = (i - (Kathod.size1 - 1) / 2) * Kathod.step;
	inout = new float* [Kathod.size2];		//Матрица, содержащая в себе информацию, является точка внутренней или внешней для катода. Заполняется функцией.
	for (int i = 0; i < Kathod.size2; i++)
		inout[i] = new float[Kathod.size1];
	u = new float[Kathod.size1 * Kathod.size2];		//Матрица, содержащая значение потенциала в точке. Заполняется функцией.
	T = new float[Kathod.size1 * Kathod.size2];
	Table = new float [35000];
	Tablereader(Table);
	mesh(inout, Kathod, 0, 1);			//Получене внутренних,внещних точек.
	for (int i = 0; i < Kathod.size2; i++) {						//Запись граничных условий в матрицу для потенциала (стоит переработать)
		for (int j = 0; j < Kathod.size1; j++) {
			T[i * Kathod.size1 + j] = inout[i][j] * 1600 * c / l;
			u[i * Kathod.size1 + j] = inout[i][j] * ((border[1][1] - border[1][0]) / (Kathod.size2 - 1) * i + border[1][0]);
			u[j] = border[1][0];
		}
	}
	for (int i = 0; i < Kathod.size2; i++) {
		std::cout << i << "\t";
		for (int j = 0; j < Kathod.size1; j++)
			std::cout << inout[i][j];
		std::cout << std::endl;
	}
	//int sta = clock();
	//int sta = time(NULL);
	solveall(inout, Table, T, u, border[1][2], xcent, x, y, Kathod, s1, s2, DirName);
	//int end = clock();
	//int end = time(NULL);
	//std::cout << "Время работы: " << (end - sta) / CLOCKS_PER_SEC << std::endl;
	//std::cout << "Время работы: " << (end - sta) << std::endl;
	derivative(u, Kathod.size1, Kathod.size2, Kathod.step, DirName);											//Вычисление производной на оси симетрии.
	for (int i = 0; i < Kathod.size2; i++) {
		delete[] inout[i];
	}
	delete[] Table;
	delete[] x;
	delete[] y;
	delete[] u;
	delete[] xcent;
	delete[] T;

	return 0;
}

//Функция, для разбиения области на точки.
__host__ void mesh(float** inout, struct Kathodparam K, float s1, float s2) {
	int size1, size2, siv, sir, sik;
	size1 = K.size1;
	size2 = K.size2;
	siv = K.siv;
	sir = K.sir;
	sik = K.sik;
	for (int j = 0; j < size1; j++) {										//Заполнение певой строки.
		if (j < (size1 - sik - 1) / 2 || j >(size1 + sik - 1) / 2)
			inout[0][j] = s1;
		else
			inout[0][j] = s2;
	}
	int A, B, C, minarg = 0, count = 0;
	int j = (size1 - 1) / 2 - sik / 2;										//Первая точка в катоде			
	A = size2 - siv - sir - 1;												//Коэффициенты для метода Брезенхема
	B = sir - sik / 2;
	C = -A * j;
	for (int i = 0; i < size2 - 1; i++) {
		if (i < size2 - siv - 1) {
			if (i - A < (size1 - 1) / 2 - j) {
				float rad[3];
				for (int k = 0; k < 3; k++) {
					if (i < A)
					{
						rad[k] = abs((A * (j + k - 0.5) + B * (i + 0.5) + C) / (sqrt(pow(A, 2) + pow(B, 2))));
					}
					else {
						rad[k] = abs(sir - sqrt(pow(((size1 - 1) / 2 - (j + k - 0.5)), 2) + pow((A - (i + 0.5)), 2)));
					}
					if ((k > 0) && (rad[k] < rad[k - 1]))
						minarg = k;
				}
				j = j - 1 + minarg;
			}
			else {
				if (i - A > (size1 - 1) / 2 - j)
					j -= 1;
				j = j - count - 1;
				while (inout[i - count][j] == s1)
					count += 1;
				j += count + 1;
			}
			for (int l = 0; l < size1; l++)
				if (l >= j && l < size1 - j)
					inout[i + 1][l] = s2;		//Подстановка коэффициентов (возможно, удобнее сделать s1 внешние и 1 внутренние0
				else
					inout[i + 1][l] = s1;		//
		}
		else {
			for (int l = 0; l < size1; l++)
				inout[i + 1][l] = s1;			//
		}
	}
}
//Основная вычисляющая функция
__host__ void solveall(float** inout, float* Table, float* T, float* u, float border, float* xcent, float* x, float* y, struct Kathodparam K, float s1, float s2, std::string DirName) {
	float* u0, * u1, * ax, * ay, * ar, * E1, * E2, * delt, YU, YT, Y, epsilon, sin1, cos1, dif, dit, buf = -1;
	float l, lambda, c, e, r0, t0, rho, k, f, koeflaplas, tauU, tauT;
	float* T_prev, * T_new, *T0;
	int* L, count = 0;
	int Tmin, Tmax, Emin, Emax, deltsum;
	int size1, size2, sir, siv, mid1, mid2;
	float step, R, r, eps;
	size1 = K.size1;
	size2 = K.size2;
	sir = K.sir;
	siv = K.siv;
	step = K.step;
	R = K.R;
	r = K.r;
	eps = K.eps;
	mid1 = K.mid1;
	mid2 = K.mid2;
	Tmin = 300;
	Tmax = 2000;
	Emin = 10;
	Emax = 10000000;
	t0 = 1;
	r0 = 1e-5;
	rho = 2330;
	l = 164000;
	lambda = 149;
	c = 678;
	e = 1.602 * 1e-19;
	epsilon = 0.004;											//Параметр
	k = (lambda) / (rho * c);
	koeflaplas = (t0 * lambda / (c * rho * r0 * r0));
	std::cout <<"koef "<<  koeflaplas << std::endl;
	tauU = pow(step, 2) / (5 * s2);
	tauT = pow(step, 2) / (5 * koeflaplas);
	std::cout <<"tauT "<< tauT << std::endl;
	f = 1 / (c * rho);
	u0 = new float[size1 * size2];						//Копии потенциала
	u1 = new float[size1 * size2];
	L = new int[mid1];								//Первые внешние точки в каждом столбце - внешняя граница.
	delt = new float[mid1 * size2];					//Коэффициент при дельта-фунции (столбец)
	YU = tauU / pow(step, 2);								//Параметр
	YT = tauT / pow(step, 2);								//Параметр
	dit = 0.003;										//Вспомогательные переменные и флаги
	dif = 2 * dit;
	buf = -1;
	ax = new float[(size1 - 1) * size2];		//Массивы параметров для учета разрывного коэффициента
	ay = new float[size1 * (size2 - 1)];		//
	ar = new float[size1 - 1];			// 
	E1 = new float[mid1 * size2];
	E2 = new float[mid1 * size2];
	T_prev = new float[mid1 * size2];
	T_new = new float[mid1 * size2];
	T0 = new float[mid1 * size2];
	for (int i = 0; i < size2; i++) {
		for (int j = 0; j < size1 - 1; j++) {
			float k1, k2;
			k1 = inout[i][j] * (s2 - s1) + s1;
			k2 = inout[i][j + 1] * (s2 - s1) + s1;
			ax[i * (size1 - 1) + j] = 2 * k1 * k2 / (k1 + k2);
		}
	}
	for (int i = 0; i < size2 - 1; i++) {
		for (int j = 0; j < size1; j++) {
			float k1, k2;
			k1 = inout[i][j] * (s2 - s1) + s1;
			k2 = inout[i + 1][j] * (s2 - s1) + s1;
			ay[i * size1 + j] = 2 * k1 * k2 / (k1 + k2);
		}
	}
	for (int i = 0; i < size1 - 1; i++) {
		ar[i] = (xcent[i] + xcent[i + 1]) / 2;
	}
	for (int i = 0; i < size2; i++) {
		for (int j = 0; j < mid1; j++) {
			u0[i * size1 + j] = u[i * size1 + j];
			u1[i * size1 + j] = u[i * size1 + j];
		}
	}
	for (int i = 0; i < size2; i++) {
		for (int j = 0; j < mid1; j++) {
			if (j < mid2 - 5)
				delt[i * mid1 + j] = 0;
			else
				delt[i * mid1 + j] = 1;
		}
	}
	sin1 = sin(M_PI / 4);
	cos1 = cos(M_PI / 4);
	int Lcount1 = 0, Lcount2 = 0;
	for (int j = 0; j < mid1; j++) {				//Крайние точки границы
		//int j = 2;
		int i = 1;
		while (inout[i][j] != 0)
			i++;
		L[j] = i;
		if (j >= K.indent) {
			if (L[j] == L[j - 1]) {
				Lcount1++;
			}
			else {
				Lcount2 += L[j] - L[j - 1] - 1;
			}
		}
	}
	//Координаты всех граничных точек выносятся в два массива...
	int borderfirstsize = (Lcount1 + Lcount2 - 1) + (mid1 - K.indent - 2) + (size2 - siv - 2);
	int bordersecondsize = mid1 - K.indent - Lcount1 + 3;
	//std::cout << "1: " << borderfirstsize << " ,2: " << bordersecondsize << std::endl;
	int* borderfirst = new int[4 * borderfirstsize];
	int* bordersecond = new int[6 * bordersecondsize];
	for (int i = 0; i < borderfirstsize; i++) {
		if (i < (mid1 - K.indent - 2)) {
			borderfirst[4 * i] = i + K.indent + 1;
			borderfirst[4 * i + 1] = 0;
			borderfirst[4 * i + 2] = i + K.indent + 1;
			borderfirst[4 * i + 3] = 1;
		}
		else if (i < (mid1 - K.indent - 2) + (size2 - siv - 2)) {
			borderfirst[4 * i] = mid2;
			borderfirst[4 * i + 1] = i - (mid1 - K.indent - 3);
			borderfirst[4 * i + 2] = mid2 - 1;
			borderfirst[4 * i + 3] = i - (mid1 - K.indent - 3);
		}
	}
	int i = (mid1 - K.indent - 2) + (size2 - siv - 2);
	int j = mid2 - 1;
	int i2 = 0;
	std::cout << j << " " << i << std::endl;
	bordersecond[6 * i2] = K.indent;
	bordersecond[6 * i2 + 1] = 0;
	bordersecond[6 * i2 + 2] = K.indent;
	bordersecond[6 * i2 + 3] = 1;
	bordersecond[6 * i2 + 4] = -1;
	bordersecond[6 * i2 + 5] = -1;
	i2 += 1;
	bordersecond[6 * i2] = mid2;
	bordersecond[6 * i2 + 1] = 0;
	bordersecond[6 * i2 + 2] = mid2;
	bordersecond[6 * i2 + 3] = 1;
	bordersecond[6 * i2 + 4] = -1;
	bordersecond[6 * i2 + 5] = -1;
	i2 += 1;
	bordersecond[6 * i2] = mid2;
	bordersecond[6 * i2 + 1] = size2 - siv - 1;
	bordersecond[6 * i2 + 2] = mid2;
	bordersecond[6 * i2 + 3] = size2 - siv - 2;
	bordersecond[6 * i2 + 4] = -1;
	bordersecond[6 * i2 + 5] = -1;
	i2 += 1;
	while (j >= K.indent)
	{
		int var = L[j] - L[j - 1];
		if (var == 0) {
			borderfirst[4 * i] = j;
			borderfirst[4 * i + 1] = L[j] - 1;
			borderfirst[4 * i + 2] = j;
			borderfirst[4 * i + 3] = L[j] - 2;
			i += 1;
			j -= 1;
		}
		else {
			bordersecond[6 * i2] = j;
			bordersecond[6 * i2 + 1] = L[j] - 1;
			bordersecond[6 * i2 + 2] = j + 1;
			bordersecond[6 * i2 + 3] = L[j] - 1;
			bordersecond[6 * i2 + 4] = j;
			bordersecond[6 * i2 + 5] = L[j] - 2;
			i2 += 1;
			int count = 1;
			while (count < var) {
				borderfirst[4 * i] = j;
				borderfirst[4 * i + 1] = L[j] - count - 1;
				borderfirst[4 * i + 2] = j + 1;
				borderfirst[4 * i + 3] = L[j] - count - 1;
				i += 1;
				count += 1;
			}
			j -= 1;
		}
	}
	//Границы записаны
	/*for (int i = 0; i < borderfirstsize; i++) {
		std::cout << "x: " << borderfirst[i * 4] << " y: " << borderfirst[i * 4 + 1] << std::endl;
	}
	for (int i = 0; i < bordersecondsize; i++) {
		std::cout << "x: " << bordersecond[i * 6] << " y: " << bordersecond[i * 6 + 1] << std::endl;
	}*/
	while (dif > dit) //Вычисление потенциала
	{
		count++;
		float buf1 = 0;
		float D = 0;
//#pragma omp parallel for schedule(static)
		for (int i = 1; i < size2 - 1; i++) {
			for (int j = 1; j < mid1 - 1; j++) {
				u0[i * size1 + j] =
					(u[i * size1 + j + 1] * ax[i * (size1 - 1) + j] * ar[j] + u[i * size1 + j - 1] * ax[i * (size1 - 1) + j - 1] * ar[j - 1]) * YU / xcent[j] +
					(u[(i + 1) * size1 + j] * ay[i * size1 + j] + u[(i - 1) * size1 + j] * ay[(i - 1) * size1 + j]) * YU +
					u[i * size1 + j] * (1 - (ax[i * (size1 - 1) + j] * ar[j] / xcent[j] + ax[i * (size1 - 1) + j - 1] * ar[j - 1] / xcent[j] + ay[i * size1 + j] + ay[(i - 1) * size1 + j]) * YU)
					- delt[i * mid1 + j] * (((exp(-(pow((i - L[j]) * step / epsilon, 2)))) / epsilon) * (inout[i][j] * s2 * 16 * tauU));
			}
			u0[i * size1] = (u0[i * size1 + 1] - step * border);			//Граничное условие.
			u0[i * size1 + mid1 - 1] = u0[i * size1 + mid1 - 2];
		}
		for (int i = 0; i < size2; i++) {
			for (int j = 0; j < mid1; j++) {
				u[i * size1 + j] = u0[i * size1 + j];
				u[(i + 1) * size1 - 1 - j] = u0[i * size1 + j];
				if (count % 1000 == 0) {
					buf1 = u[i * size1 + j] - u1[i * size1 + j];
					if (buf < buf1)
						buf = buf1;
					u1[i * size1 + j] = u[i * size1 + j];
					u1[(i + 1) * size1 - 1 - j] = u[i * size1 + j];
				}
			}
		}
		if (count % 1000 == 0) {
			dif = buf;
			buf = 0;
			std::ofstream out(get_name(DirName, "deltU"), std::ios::app);
			if (out.is_open())
			{
				out << dif << "\t";
			}
			out.close();
			output_vtk_binary(u, size1 - 1, size2 - 1, x, y, get_name(DirName, "densU", count));
			derivative(u, size1, size2, step, DirName);
		}
	}							//Потенциал посчитан
	std::cout << count << std::endl;

	//Напряженность
	for (int j = 0; j < mid1; j++) {
		E1[j] = 0;
		E1[(size2 - 1) * mid1 + j] = 0;
		E2[j] = 0;
		E2[(size2 - 1) * mid1 + j] = 0;
	}
	for (int i = 1; i < size2 - 1; i++) {
		E1[i * mid1] = 0;
		E2[i * mid1] = 0;
		for (int j = 1; j < mid1; j++) {
			E1[i * mid1 + j] = sqrt((pow(u[(i + 1) * size1 + j] - u[(i - 1) * size1 + j], 2) + pow(u[i * size1 + j] - u[i * size1 + j], 2)) / (4 * step * step));
			E2[i * mid1 + j] = abs(u[(i + 1) * size1 + j] - u[i * size1 + j]) / (2 * step);
		}
	}

	count = 0;
	dit = 0.003;										//Вспомогательные переменные и флаги
	dif = 2 * dit;
	buf = 0;
	//Массивы для видеокарты
	float* dev_t_prev = 0;
	float* dev_t_new = 0;
	float* dev_ar = 0;
	float* dev_xcent = 0;
	float* dev_E1 = 0;
	float* dev_E2 = 0;
	float* dev_Table = 0;
	int* dev_L = 0;
	int* dev_b_first = 0;
	int* dev_b_second = 0;
	

	for (int i = 0; i < size2; i++) {
		for (int j = 0; j < mid1; j++) {
			T_new[i * mid1 + j] = T[i * size1 + j];
			T_prev[i * mid1 + j] = T[i * size1 + j];
			T0[i * mid1 + j] = T[i * size1 + j];
		}
	}
	//Функция чтения таблицы переписана под одномерный массив, проверить на ошибки при работе.

	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess)
	{
		std::cerr << "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?" << std::endl;
	}

	cudaStatus = cudaMalloc((void**)&dev_t_prev, mid1 * size2 * sizeof(float));
	if (cudaStatus != cudaSuccess)
	{
		std::cerr << "cudaMalloc failed!" << std::endl;
	}

	cudaStatus = cudaMalloc((void**)&dev_t_new, mid1 * size2 * sizeof(float));
	if (cudaStatus != cudaSuccess)
	{
		std::cerr << "cudaMalloc failed!" << std::endl;
	}

	cudaStatus = cudaMalloc((void**)&dev_ar, (size1 - 1) * sizeof(float));
	if (cudaStatus != cudaSuccess)
	{
		std::cerr << "cudaMalloc failed!" << std::endl;
	}

	cudaStatus = cudaMalloc((void**)&dev_xcent, size1 * sizeof(float));
	if (cudaStatus != cudaSuccess)
	{
		std::cerr << "cudaMalloc failed!" << std::endl;
	}

	cudaStatus = cudaMalloc((void**)&dev_E1, mid1 * size2 * sizeof(float));
	if (cudaStatus != cudaSuccess)
	{
		std::cerr << "cudaMalloc failed!" << std::endl;
	}

	cudaStatus = cudaMalloc((void**)&dev_E2, mid1 * size2 * sizeof(float));
	if (cudaStatus != cudaSuccess)
	{
		std::cerr << "cudaMalloc failed!" << std::endl;
	}

	cudaStatus = cudaMalloc((void**)&dev_L, mid1 * size2 * sizeof(int));
	if (cudaStatus != cudaSuccess)
	{
		std::cerr << "cudaMalloc failed!" << std::endl;
	}

	cudaStatus = cudaMalloc((void**)&dev_Table, 35000 * sizeof(float));
	if (cudaStatus != cudaSuccess)
	{
		std::cerr << "cudaMalloc failed!" << std::endl;
	}

	cudaStatus = cudaMalloc((void**)&dev_b_first, 4 * borderfirstsize * sizeof(int));
	if (cudaStatus != cudaSuccess)
	{
		std::cerr << "cudaMalloc failed!" << std::endl;
	}

	cudaStatus = cudaMalloc((void**)&dev_b_second, 6 * bordersecondsize * sizeof(int));
	if (cudaStatus != cudaSuccess)
	{
		std::cerr << "cudaMalloc failed!" << std::endl;
	}

	cudaStatus = cudaMemcpy(dev_t_prev, T_prev, mid1 * size2 * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		std::cerr << "cudaMemcpy failed!" << std::endl;
	}

	cudaStatus = cudaMemcpy(dev_ar, ar, (size1 - 1) * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		std::cerr << "cudaMemcpy failed!" << std::endl;
	}
	cudaStatus = cudaMemcpy(dev_t_new, T_new, mid1 * size2 * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		std::cerr << "cudaMemcpy failed!" << std::endl;
	}

	cudaStatus = cudaMemcpy(dev_xcent, xcent, size1 * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		std::cerr << "cudaMemcpy failed!" << std::endl;
	}

	cudaStatus = cudaMemcpy(dev_E1, E1, mid1 * size1 * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		std::cerr << "cudaMemcpy failed!" << std::endl;
	}

	cudaStatus = cudaMemcpy(dev_E2, E2, mid1 * size1 * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		std::cerr << "cudaMemcpy failed!" << std::endl;
	}

	cudaStatus = cudaMemcpy(dev_L, L, size1 * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		std::cerr << "cudaMemcpy failed!" << std::endl;
	}

	cudaStatus = cudaMemcpy(dev_Table, Table, 35000 * sizeof(float), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		std::cerr << "cudaMemcpy failed!" << std::endl;
	}

	cudaStatus = cudaMemcpy(dev_b_first, borderfirst, 4 * borderfirstsize * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		std::cerr << "cudaMemcpy failed!" << std::endl;
	}

	cudaStatus = cudaMemcpy(dev_b_second, bordersecond, 6 * bordersecondsize * sizeof(int), cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess)
	{
		std::cerr << "cudaMemcpy failed!" << std::endl;
	}

	dim3 threads2dbase(BLOCK_SIZE, 2*BLOCK_SIZE);
	dim3 blocks2dbase((mid1 + 1) / threads2dbase.x + 1, (size2 + 1) / threads2dbase.y + 1);//НАДО БРАТЬ СОРАЗМЕРНЫЕ БЛОКУ СЕТКИ
	std::cout << blocks2dbase.x << " " << blocks2dbase.y << std::endl;
	dim3 threads2dfirst(20, 1);
	dim3 blocks2dfirst((borderfirstsize + 1) / threads2dfirst.x + 1, 1);//НАДО БРАТЬ СОРАЗМЕРНЫЕ БЛОКУ СЕТКИ
	std::cout << blocks2dfirst.x << " " << blocks2dfirst.y << std::endl;
	dim3 threads2dsecond(10, 1);
	dim3 blocks2dsecond((bordersecondsize + 1) / threads2dsecond.x + 1, 1);//НАДО БРАТЬ СОРАЗМЕРНЫЕ БЛОКУ СЕТКИ
	std::cout << blocks2dsecond.x << " " << blocks2dsecond.y << std::endl;

	count = 0;
	float* temp;

	output_vtk_binary(T, size1 - 1, size2 - 1, x, y, get_name(DirName, "densT", count));
	int sta = time(NULL);
	while (count < 4000000) {//Вычисление ур-ия для потенциала
		float buf1 = 0;
		count++;
		/*basefunc <<< blocks2dbase, threads2dbase >>> (dev_t_prev, dev_t_new, dev_ar, dev_xcent, dev_E1, dev_L, YT, koeflaplas, s2, tauT, t0, l, rho, c, r0, K);

		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess)
		{
			std::cerr << "cudaDeviceSynchronize returned error code after launching addKernel!  basefunc" << cudaStatus << std::endl;
		}*/

		borderfunc1 <<< blocks2dfirst, threads2dfirst >>> (dev_t_prev, dev_t_new, dev_E2, dev_Table, dev_b_first, borderfirstsize, t0, l, c, r0, lambda, K);

		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess)
		{
			std::cerr << "cudaDeviceSynchronize returned error code after launching addKernel! borderfirst" << cudaStatus << std::endl;
		}

		borderfunc2 <<< blocks2dsecond, threads2dsecond >>> (dev_t_prev, dev_t_new, dev_E2, dev_Table, dev_b_second, bordersecondsize, l, cos1, sin1, c, r0, lambda, K);

		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess)
		{
			std::cerr << "cudaDeviceSynchronize returned error code after launching addKernel! bordersecond" << cudaStatus << std::endl;
		}

		temp = dev_t_new;
		dev_t_new = dev_t_prev;
		dev_t_prev = temp;

		//output_vtk_binary(T, size1 - 1, size2 - 1, x, y, get_name(DirName, "densT", count));
		/*for (int i = 0; i < size2; i++) {
			for (int j = 0; j < mid1; j++) {
				T[i * size1 + j] = T0[i * mid1 + j];
				T[(i + 1) * size1 - 1 - j] = T0[i * mid1 + j];
				if (count % 4000000 == 0) {
					buf1 = T[i * size1 + j] - T1[i * mid1 + j];
					if (buf < buf1)
						buf = buf1;
					T1[i * mid1 + j] = T0[i * mid1 + j];
				}
			}
		}*/
		//output_vtk_binary(t, size1 - 1, size2 - 1, x, y, get_name(DirName, "densT", count));
		if (count % 40000 == 0) {

			cudaStatus = cudaMemcpy(T_prev, dev_t_prev, mid1 * size2 * sizeof(float), cudaMemcpyDeviceToHost);
			if (cudaStatus != cudaSuccess)
			{
				std::cerr << "cudaMemcpy failed!" << std::endl;
			}
			for (int i = 0; i < size2; i++) {
				for (int j = 0; j < mid1; j++) {
					T[i * size1 + j] = T_prev[i * mid1 + j];
					T[(i + 1) * size1 - 1 - j] = T_prev[i * mid1 + j];
					buf1 = T0[i * mid1 + j] - T[i * size1 + j];
					if (buf < buf1)
						buf = buf1;
					T0[i * mid1 + j] = T[i * size1 + j];
				}
			}
			dif = buf;
			std::cout << count << " " << dif << std::endl;
			buf = 0;
			std::ofstream out(get_name(DirName, "deltT"), std::ios::app);
			if (out.is_open())
			{
				out << dif << "\t";
			}
			out.close();
			output_vtk_binary(T, size1 - 1, size2 - 1, x, y, get_name(DirName, "densT", count));
		}
	}
	int end = time(NULL);
	std::cout << "Время работы: " << (end - sta) << std::endl;
	std::cout << count << std::endl;
	delete[] L;
	delete[] T_new;
	delete[] T_prev;
	delete[] T0;
	delete[] u0;
	delete[] u1;
	delete[] delt;
	delete[] ax;
	delete[] ay;
	delete[] ar;
	delete[] E1;
	delete[] E2;
	delete[] borderfirst;
	delete[] bordersecond;
	cudaFree(dev_t_new);
	cudaFree(dev_t_prev);
	cudaFree(dev_ar);
	cudaFree(dev_xcent);
	cudaFree(dev_E1);
	cudaFree(dev_E2);
	cudaFree(dev_L);
	cudaFree(dev_Table);
	cudaFree(dev_b_first);
	cudaFree(dev_b_second);
}

__global__ void borderfunc1(float* T, float* T0, float* E2, float* Table, int *borderfirst, int borderfirstsize, float t0, float l, float c, float r0, float lambda, struct Kathodparam K) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (i >= 0 && i < borderfirstsize) {
		if (borderfirst[4 * i + 1] == 0) {
			T0[borderfirst[4 * i]] = T0[K.mid1 + borderfirst[4 * i]];		//Условие на нижней границе - производная равна 0.
		}
		else if (borderfirst[4 * i] == K.mid2) {
			T0[borderfirst[4 * i + 1] * K.mid1 + K.mid2] = T0[borderfirst[4 * i + 1] * K.mid1 + K.mid2 - 1];		//Условие на оси
		}
		else if (borderfirst[4 * i] <= K.mid2 - K.sir) {
			T0[borderfirst[4 * i + 1] * K.mid1 + borderfirst[4 * i]] =
				T0[borderfirst[4 * i + 3] * K.mid1 + borderfirst[4 * i + 2]] -
				K.step * (c * r0) / (10000 * l * lambda) *
				jem(Table, r0, l * T[borderfirst[4 * i + 1] * K.mid1 + borderfirst[4 * i]] / c, E2[borderfirst[4 * i + 1] * K.mid1 + borderfirst[4 * i]]) *
				E(T[borderfirst[4 * i + 1] * K.mid1 + borderfirst[4 * i]]);
		}																	//Эмиссия на прямых участках
		else {
			T0[borderfirst[4 * i + 1] * K.mid1 + borderfirst[4 * i]] = T0[borderfirst[4 * i + 3] * K.mid1 + borderfirst[4 * i + 2]];	//Условие на боковой поверхности
		}
	}
}

__global__ void borderfunc2(float* T, float* T0, float* E2, float* Table, int *bordersecond, int bordersecondsize, float l, float cos1, float sin1, float c, float r0, float lambda, struct Kathodparam K) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	if (bordersecond[6 * i + 1] == 0) {					//Условие на оси/нижней границе
		T0[bordersecond[6 * i]] = T0[K.mid1 + bordersecond[6 * i]];
	}
	else if (bordersecond[6 * i] <= K.mid2) {
		T0[bordersecond[6 * i + 1] * K.mid1 + bordersecond[6 * i]] =
			T0[bordersecond[6 * i + 3] * K.mid1 + bordersecond[6 * i + 2]] - 
			K.step * (c * r0) / (10000 * l * lambda) * 
			jem(Table, r0, l * T[bordersecond[6 * i + 1] * K.mid1 + bordersecond[6 * i]] / c, E2[bordersecond[6 * i + 1] * K.mid1 + bordersecond[6 * i]]) *
			E(T[bordersecond[6 * i + 1] * K.mid1 + bordersecond[6 * i]]);		//Эмиссия на оси и углах
	}
	else {
		T0[bordersecond[6 * i + 1] * K.mid1 + bordersecond[6 * i]] = 
			(T0[bordersecond[6 * i + 3] * K.mid1 + bordersecond[6 * i + 2]] * cos1 +
				T0[bordersecond[6 * i + 5] * K.mid1 + bordersecond[6 * i + 4]] * sin1) / (cos1 + sin1);
	}						//Условие на углах
}

__global__ void basefunc(float* T, float* T0, float* ar, float* xcent, float *E1, int* L, float YT, float koeflaplas, float s2,float tauT, float t0, float l, float rho, float c, float r0, struct Kathodparam K){
	int j = threadIdx.x + blockIdx.x * blockDim.x;
	int i = threadIdx.y + blockIdx.y * blockDim.y;
	if (j > 3 && j < K.mid2) {									//под номером четыре вторая точка внутри катода, до середины ( не включая)
		if (i > 0 && i < L[j-1]) {									//Обход внутренних точек катода
			T0[i * K.mid1 + j] =
				(((T[i * K.mid1 + j - 1] - T[i * K.mid1 + j]) * ar[j - 1] - (T[i * K.mid1 + j] - T[i * K.mid1 + j + 1]) * ar[j]) * xcent[j] +
				((T[(i + 1) * K.mid1 + j] - T[i * K.mid1 + j]) - (T[i * K.mid1 + j] - T[(i - 1) * K.mid1 + j]))) * YT * koeflaplas + T[i * K.mid1 + j];
			+s2 * pow(E1[j * K.mid1 + i], 2) * (tauT * t0) / (l * rho);
		}
	}
}


__host__ void Tablereader(float* Table) { //Проверять в новом проектк источник!!!!
	std::string line;
	std::ifstream in("D:\\ProjectM2\\Table1.txt"); // окрываем файл для чтения  
	//std::ifstream in("Table1.txt"); // окрываем файл для чтения
	if (in.is_open())
	{
		int count = 0;
		while (getline(in, line))
		{
			int count = 0;
			int i = 0, j = 0;
			while (i <= line.find_last_of(" "))
			{
				i = line.find(" ", i + 1);
				Table[count] = std::stod(line.substr(j, i - j));
				count++;
				j = i + 1;
			}
		}
	}
	in.close();     // закрываем файл
}

__device__ float jem(float* Table, float r0, float T, float E) {
	float ee, tt, j = 0;
	ee = E / (10 * r0);
	tt = T;
	if (ee < 10)
		ee = 10;
	if (tt < 300)
		tt = 300;
	if (ee > 10e7)
		ee = 10e7 - 1;
	if (tt > 2000)
		tt = 2000 - 1;
	int k[4] = { 0 };
	float x0, y0;
	x0 = (tt - 300) / 50;
	y0 = (ee - 10) / 9999.99;
	k[0] = floor(x0);
	k[1] = floor(y0);
	k[2] = k[0] + 1;
	k[3] = k[1] + 1;
	j = Table[k[0] +k [1]*1000] * (k[2] - x0) * (k[3] - y0) + Table[k[2] + k[1] *1000] * (x0 - k[0]) * (k[3] - y0) +
		Table[k[0] + k[3] * 1000] * (k[2] - x0) * (y0 - k[1]) + Table[k[2] + k[3] * 1000] * (x0 - k[0]) * (y0 - k[1]);
	//std::cout << tt << " " << E <<" "<<r0<<" " << " " << j << std::endl;
	return(j);
}

__device__ float Ef(float* E1, float* E2, int mid1, int size2, int sir, int siv, int i, int j) {
	float cos, sin, x, y;
	x = abs(mid1 - 1 - j);
	y = abs(size2 - sir - siv - 1 - i);
	cos = x / sqrt(pow(x, 2) + pow(y, 2));
	sin = y / sqrt(pow(x, 2) + pow(y, 2));
	return((pow(E1[i * mid1 + j] * cos, 2) + pow(E2[i * mid1 + j] * sin, 2)));
}

__device__ float E(float T) {
	float E;
	if (T < 0.2129061)
		E = -0.0589529 * T / tan(14.6137 * T);
	else
		E = 4.42871 + 0.0417038 * T + (-21, 8518 + 0.25058 * T) / (4.92306 + pow((-1 + 9.30338 * T), 3.48481));
	return (E);
}

//float Dcoef(float** Table, float* u, float* T, int* L, int size1, int step, int s2, float r0) {
//	float D = 0;
//	for (int j = (size1 - 1) / 2 - 5; j < (size1 - 1) / 2; j++) {
//		for (int i = L[j - 1] - 1; i < L[j]; i++) {
//			D += jem(Table, r0, T[i * size1 + j], sqrt(pow(u[(i + 1) * size1 + j] - u[i * size1 + j], 2) + pow(u[i * size1 + j] - u[i * size1 + j + 1], 2)) / step) / (s2 * sqrt(pow(u[i * size1 + j] - u[(i - 1) * size1 + j], 2) + pow(u[i * size1 + j] - u[i * size1 + j + 1], 2)) / step);
//		}
//	}
//	D += D;
//	int j = (size1 - 1) / 2;
//	int i = L[j] - 1;
//	D += jem(Table, r0, T[i * size1 + j], sqrt(pow(u[(i + 1) * size1 + j] - u[i * size1 + j], 2) + pow(u[i * size1 + j] - u[i * size1 + j + 1], 2)) / step) / (s2 * sqrt(pow(u[i * size1 + j] - u[(i - 1) * size1 + j], 2) + pow(u[i * size1 + j] - u[i * size1 + j + 1], 2)) / step);
//	D = D / 11;
//	return D;
//}

__host__ void derivative(float* u, int size1, int size2, float step, std::string DirName)
{
	float der;
	int mid;
	mid = (size1 + 1) / 2;
	for (int i = 0; i < size2 - 1; i++)
	{
		der = (u[(i + 1) * size1 + mid] - u[i * size1 + mid]) / step;
		std::ofstream out(get_name(DirName, "derivative"), std::ios::app);
		if (out.is_open())
		{
			out << i * step << "\t" << der << std::endl;
		}
		out.close();
	}
}

__host__ std::string get_dir_name()  //Менять для СКП
{
	std::string DirTimeName;
	/*time_t rawtime;
	struct tm timeinfo;
	time(&rawtime);
	localtime_s(&timeinfo ,&rawtime);
	DirTimeName = std::to_string((&timeinfo).tm_hour) + "_" + std::to_string((&timeinfo).tm_min) + "___" + std::to_string((&timeinfo).tm_mday) + "_" + std::to_string((&timeinfo).tm_mon + 1);
	*/DirTimeName = "100";
	return DirTimeName;
}

__host__ std::string get_name(std::string name, int k)
{
	std::string st2;
	st2 = name;
	std::ostringstream s;
	s << k;
	st2 += s.str();
	st2 += ".vtk";
	return (st2);
}

__host__ std::string get_name(std::string DirName, std::string name)
{
	std::string st2;
	st2 = name;
	st2 += ".txt";
	return (DirName + "\\" + st2);
}

__host__ std::string get_name(std::string DirName, std::string name, int k)
{
	std::string st2;
	st2 = name;
	std::ostringstream s;
	s << k;
	st2 += s.str();
	st2 += ".vtk";
	return (DirName + "\\" + st2);
}

template <typename T>
__host__  void SwapEnd(T& var)
{
	char* varArray = reinterpret_cast<char*>(&var);
	for (long i = 0; i < static_cast<long>(sizeof(var) / 2); i++)
		std::swap(varArray[sizeof(var) - 1 - i], varArray[i]);
}

__host__  void output_vtk_binary(float* u, int n1, int n2, float* x, float* y, std::string name)
{

	int arr_size = (n1 + 1) * (n2 + 1);
	float* my_u = new float[arr_size];
	for (int j = 0; j <= n2; j++)
	{
		for (int i = 0; i <= n1; i++)
		{
			my_u[j * (n1 + 1) + i] = u[j * (n1 + 1) + i];
			SwapEnd(my_u[j * (n1 + 1) + i]);
		}
	}



	std::ofstream fileD;

	fileD.open(name.c_str(), std::ios::out | std::ios::trunc | std::ios::binary);
	fileD << "# vtk DataFile Version 2.0" << "\n";
	fileD << "PRESS" << "\n";
	fileD << "BINARY" << "\n";
	fileD << "DATASET STRUCTURED_GRID" << std::endl;
	fileD << "DIMENSIONS " << n1 + 1 << " " << n2 + 1 << " " << "1" << std::endl;
	fileD << "POINTS " << arr_size << " float" << std::endl;
	float tt1, tt2, tt3 = 0;
	SwapEnd(tt3);
	for (int j = 0; j <= n2; j++)
	{
		for (int i = 0; i <= n1; i++)
		{
			tt1 = x[i];
			tt2 = y[j];
			SwapEnd(tt1);
			SwapEnd(tt2);

			fileD.write((char*)&tt1, sizeof(float));
			fileD.write((char*)&tt2, sizeof(float));
			fileD.write((char*)&tt3, sizeof(float));

		}
	}
	fileD << "POINT_DATA " << arr_size << std::endl;
	fileD << "SCALARS phi float 1" << std::endl;
	fileD << "LOOKUP_TABLE default" << std::endl;

	for (int j = 0; j <= n2; j++)
	{
		for (int i = 0; i <= n1; i++)
		{
			fileD.write((char*)&my_u[j * (n1 + 1) + i], sizeof(float));
		}
	}


	fileD.close();


	delete[] my_u;
}