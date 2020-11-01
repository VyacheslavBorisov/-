// ConsoleApplication2.cpp : Этот файл содержит функцию "main". Здесь начинается и заканчивается выполнение программы.
//
#define _USE_MATH_DEFINES

#include <iostream>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <cstring>
#include <cmath>
#include <ctime>
#include <omp.h>
#include <string>




void mesh(double** inout, int size1, int size2, int siv, int sir, int sik,double s1, double s2);
void solveall(double** inout, double **Table, double* T, double* u, double border, int size1, int size2, double step,double *xcent, double* x, double* y, double R, double r, double eps, int sir, int siv, std::string DirName, double s1, double s2);
void derivative(double* u, int size1, int size2, double step, std::string DirName);
double jem(double **Table, double r0, double T, double E);
double Ef(double* E1, double* E2, int mid1, int size2, int sir, int siv, int i, int j);
double E(double T);
double Dcoef(double** Table, double* u, double* T, int* L, int size1, int step, int s2, double r0);
void Tablereader(double** Table);
std::string get_dir_name();
std::string get_name(std::string name, int k);
std::string get_name(std::string DirName, std::string name);
std::string get_name(std::string DirName, std::string name, int k);
void output_vtk_binary(double* u, int n1, int n2, double* x, double* y, std::string name);


int main() {
	setlocale(LC_ALL, "Russian");
	omp_set_num_threads(1);
	double R, r, step, eps;
	double** Table;
	double** inout, * x, * y,*xcent, border[2][3], start, tau, * u, *T;
	int size1, size2, siv, sir,siR, sik;
	double s1, s2,c,l;
	l = 164000;
	c = 678;
	s1 = 0.001;		//Коэффициент вне катода
	s2 = 1000;		//Внутри катода
	std::string DirName = get_dir_name();
	system(("mkdir " + DirName).c_str());
	/*std::cout << "R: " << std::endl;
	std::cin >> R;
	std::cout << "r: " << std::endl;
	std::cin >> r;
	std::cout << "eps:" << std::endl;
	std::cin >> eps;
	std::cout << std::endl;*/
	R = 30;
	r = 3;
	eps = 3;
	border[0][0] = 1;
	border[1][0] = 4;
	border[0][1] = 1;
	border[1][1] = 0;
	border[0][2] = 2;
	border[1][2] = 0;
	start = 0;
	sir = 5;							//Шагов в радиусе.
	step = r / (sir * 100);				//Длина шага.
	siR = floor(R * sir / r);			//Шагов в большем радиусе.
	size1 = 7 + 2 * siR;				//Размер области по горизонтали в точках, нечетное число.
	size2 = floor(1 / step + 1);		//Размер области по вертикали вточках.
	siv = floor(eps / (100 * step));	//Число шагов в вакууме.
	sik = 2*siR;						//Диаметр большего основания в шагах.
	x = new double[size1];				//Массив координат по Х.
	y = new double[size2];				//Массив координат по У.
	for (int i = 0; i < size1; i++)
		x[i] = i * step;
	for (int i = 0; i < size2; i++)
		y[i] = i * step;
	xcent = new double[size1];				//Массив координат по Х.
	for (int i = 0; i < size1; i++)
		xcent[i] = (i - (size1 - 1)/2)* step;
	inout = new double* [size2];		//Матрица, содержащая в себе информацию, является точка внутренней или внешней для катода. Заполняется функцией.
	for (int i = 0; i < size2; i++)
		inout[i] = new double[size1];
	u = new double[size1 * size2];		//Матрица, содержащая значение потенциала в точке. Заполняется функцией.
	T = new double[size1 * size2];
	Table = new double* [35];
	for (int i = 0; i < 35; i++)
		Table[i] = new double[1000];
	Tablereader(Table);
	mesh(inout, size1, size2, siv, sir, sik,0,1);			//Получене внутренних,внещних точек.
	for (int i = 0; i < size2; i++) {						//Запись граничных условий в матрицу для потенциала (стоит переработать)
		for (int j = 0; j < size1; j++) {
			T[i * size1 + j] = inout[i][j]*1600*c/l;
			u[i * size1 + j] = inout[i][j] * ((border[1][1] - border[1][0]) / (size2 - 1) * i + border[1][0]);
			u[j] = border[1][0];
		}
	}
	for (int i = 0; i < size2; i++) {
		std::cout << i << "\t";
		for (int j = 0; j < size1; j++)
			std::cout << inout[i][j];
		std::cout << std::endl;
	}
	//int sta = clock();
	//int sta = time(NULL);
	solveall(inout, Table,T, u, border[1][2], size1, size2, step,xcent, x, y, R, r, eps, sir, siv, DirName, s1, s2);
	//int end = clock();
	////std::cout << "Время работы: " << (end - sta) << std::endl;
	derivative(u, size1, size2, step, DirName);											//Вычисление производной на оси симетрии.
	for (int i = 0; i < size2; i++) {
		delete[] inout[i];
	}
	for (int i = 0; i < 35; i++) {
		delete[] Table[i];
	}
	delete[] x;
	delete[] y;
	delete[] u;
	delete[] xcent;
	delete[] T;

	return 0;
}


void mesh(double** inout, int size1, int size2, int siv, int sir, int sik, double s1, double s2) {
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
				double rad[3];
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

void solveall(double** inout, double **Table, double* T, double* u, double border, int size1, int size2, double step, double* xcent, double* x, double* y, double R, double r, double eps, int sir, int siv, std::string DirName, double s1, double s2) {
	double* u0, * u1, * ax, * ay, * ar, * E1, * E2, *E3, *E4, *E5, * delt, YU, YT,Y, epsilon, sin1, cos1, dif, dit, buf = -1;
	double l, lambda, c, e , r0, t0 ,rho, k, f, koeflaplas, tauU,tauT;
	double * T0, *T1;
	int *L, count = 0, mid1, mid2;
	int Tmin, Tmax, Emin, Emax,deltsum;
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
	mid1 = (size1 + 1) / 2;								//Центр по Х
	mid2 = (size1 - 1) / 2;								//Центр по Х уменьшенный(?)
	epsilon = 0.004;											//Параметр
	k = (lambda) / (rho * c );
	koeflaplas = (t0 * lambda / (c * rho * r0 * r0));
	tauU = pow(step, 2) / (5 * s2);
	tauT = pow(step, 2) / (5 * koeflaplas);
	f = 1 / (c * rho);
	u0 = new double[size1 * size2];						//Копии потенциала
	u1 = new double[size1 * size2];
	L = new int[size1];								//Первые внешние точки в каждом столбце - внешняя граница.
	delt = new double[mid1 * size2];					//Коэффициент при дельта-фунции (столбец)
	YU = tauU / pow(step, 2);								//Параметр
	YT = tauT / pow(step, 2);								//Параметр
	dit = 0.003;										//Вспомогательные переменные и флаги
	dif = 2 * dit;
	buf = -1;
	ax = new double[(size1 - 1) * size2];		//Массивы параметров для учета разрывного коэффициента
	ay = new double[size1 * (size2 - 1)];		//
	ar = new double[size1 - 1];			// 
	E1 = new double[mid1 * size2];
	E2 = new double[mid1 * size2];
	E3 = new double[mid1 * size2];
	E4 = new double[mid1 * size2];
	E5 = new double[size1 * size2];
	T0 = new double[mid1 * size2];
	T1 = new double[mid1 * size2];
	for (int i = 0; i < size2; i++) {
		for (int j = 0; j < size1 - 1; j++) {
			double k1, k2;
			k1 = inout[i][j] * (s2 - s1) + s1;
			k2 = inout[i][j + 1] * (s2 - s1) + s1;
			ax[i * (size1 - 1) + j] = 2 * k1 * k2 / (k1 + k2);
		}
	}
	for (int i = 0; i < size2 - 1; i++) {
		for (int j = 0; j < size1; j++) {
			double k1, k2;
			k1 = inout[i][j] * (s2 - s1) + s1;
			k2 = inout[i + 1][j] * (s2 - s1) + s1;
			ay[i * size1 + j] = 2*k1*k2/(k1 + k2);
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
	/*sin = (R - r) / (sqrt(pow(R - r, 2) + pow(1 - eps - r, 2)));
	cos = -(1 - eps - r) / (sqrt(pow(R - r, 2) + pow(1 - eps - r, 2)));*/
	sin1 = sin(M_PI / 4);
	cos1 = cos(M_PI / 4);
	for (int j = 0; j < size1; j++) {
		//int j = 2;
		int i = 1;
		while (inout[i][j] != 0)
			i++;
		L[j] = i;
	}
	for (int j = 0; j < mid1; j++) {
		for (int i = 0; i < size2; i++)
		{
			T0[i * mid1 + j] = T[i * size1 + j];
		}
	}
	while (dif > dit)
	{
		count++;
		double buf1 = 0;
		double D = 0;
		#pragma omp parallel for schedule(static)
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
				if (count % 1000 == 0){
					buf1 = u[i * size1 + j] - u1[i * size1 + j];
					if (buf < buf1)
						buf = buf1;
					u1[i * size1 + j] = u[i * size1 + j];
					u1[(i + 1) * size1 - 1 - j] = u[i * size1 + j];
				}
			}
		}
		if (count % 1000 == 0) {
			//D = Dcoef(Table, u, t, L, size1, step, s2, r0);
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

	// Отсюда меняю код
	/*for (int i = 0; i < size2; i++) {
		E1[i * mid1] = 0;
		for (int j = 1; j < mid1; j++) {
			deltsum = (inout[i][j - 1] + inout[i][j]);
			if (deltsum == 0)
				deltsum = 1;
			E1[i * mid1 + j] = ((u[i * size1 + j + 1] - u[i * size1 + j]) * inout[i][j] + (u[i * size1 + j] - u[i * size1 + j - 1]) * inout[i][j - 1]) / (deltsum * step);
		}
	}
	for (int i = 1; i < size2 - 1; i++) {
		for (int j = 0; j < mid1; j++) {
			E2[j] = ((u[size1 + j] - u[j]) * inout[1][j]) / step;
			E2[(size2 - 1) * mid1 + j] = 0;
			deltsum = (inout[i + 1][j] + inout[i][j]);
			if (deltsum == 0)
				deltsum = 1;
			E2[i * mid1 + j] = ((u[(i - 1) * size1 + j] - u[i * size1 + j]) * inout[i][j] + (u[i * size1 + j] - u[(i - 1) * size1 + j]) * inout[i + 1][j]) / (deltsum * step);
		}
	}
	for (int i = 1; i < size2 - 2; i++) {
		for (int j = 0; j < mid1; j++) {
			E3[i * mid1 + j] = (u[(i + 1) * size1 + j] - u[(i)*size1 + j]) * inout[i][j] / (step);
		}
	}
	for (int i = 0; i < size2; i++) {
		for (int j = 1; j < mid1 - 1; j++) {
			E4[i * mid1 + j] = (u[i * size1 + j - 1] - u[i * size1 + j]) * inout[i][j] / (step);
		}
	}
	for (int i = 0; i < size2; i++) {
		for (int j = 0; j < mid1; j++) {
			E5[i * size1 + j] = (pow(E4[i * mid1 + j], 2) + pow(E3[i * mid1 + j], 2)); 
			E5[i * size1 + size1 - 1 - j] = (pow(E4[i * mid1 + j], 2) + pow(E3[i * mid1 + j], 2));
		}
	}
	output_vtk_binary(E5, size1 - 1, size2 - 1, x, y, get_name(DirName, "densE", 6));*/
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

	//std::cout << E5[158 * size1 + 48] << " " << E5[159 * size1 + 48] << " " << E5[160 * size1 + 49] << " " << E5[161 * size1 + 50] << std::endl;
	count = 0;
	dit = 0.003;										//Вспомогательные переменные и флаги
	dif = 2 * dit;
	buf = 0;
	for (int i = 0; i < size2; i++) {
		for (int j = 0; j < mid1; j++) {
			T0[i * mid1 + j] = T[i * size1 + j];
			T1[i * mid1 + j] = T[i * size1 + j];
		}
	}

	/*for (int i = 0; i < size2; i++) {
		for (int j = 0; j < mid1; j++) {
			T[i * size1 + j] = 1600 * c / l;
			T[i * size1  +size1 -1- j] = 1600 * c / l;
			T0[i * mid1 + j] = T[i * size1 + j];
			T1[i * mid1 + j] = T[i * size1 + j];
		}
	}*/
	//while (count < 10000000) {
	output_vtk_binary(T, size1 - 1, size2 - 1, x, y, get_name(DirName, "densT", count));
	//while (dif > dit) {
	int sta = time(NULL);
	while (count < 1) {
		double buf1 = 0;
		count++;
		#pragma omp parallel for schedule(static)
		for (int j = 4; j < mid1; j++) {									//под номером четыре вторая точка внутри катода, до середины ( не включая)
			for (int i = 1; i < L[j]; i++) {									//Обход внутренних точек катода
				T0[i * mid1 + j] =
					((T[i * size1 + j - 1] - T[i * size1 + j]) * ar[j - 1] - (T[i * size1 + j] - T[i * size1 + j + 1]) * ar[j]) * YT * xcent[j] * koeflaplas +
					((T[(i + 1) * size1 + j] - T[i * size1 + j]) - (T[i * size1 + j] - T[(i - 1) * size1 + j])) * YT * koeflaplas + T[i * size1 + j];
					+ s2 * pow(E1[j * size1 + i],2) * (tauT * t0) / (l * rho);
					
			}
		}
		for (int i = 1; i < size2; i++)
			T0[i * mid1 + mid2] = T0[i * mid1 + mid2 - 1];
		for (int j = 0; j < (size1 + 1) / 2; j++)
			T0[j] = T0[mid1 + j];					//Условие на нижней границе - производная равна 0.
		
		for (int j = 3; j < mid2 - sir; j++) {  // Подправить 
				for (int k = L[j - 1]; k < L[j] - 1; k++) {
					T0[k * mid1 + j] = T0[k * mid1 + j + 1];
				}
				T0[(L[j] - 1) * mid1 + j] = (T0[(L[j] - 1) * mid1 + j + 1] * cos1 + T0[(L[j] - 2) * mid1 + j] * sin1) / (cos1 + sin1);
		}
		for (int j = mid2 - sir; j < mid1; j++) {
			if (L[j - 1] != L[j]) {
				for (int k = L[j - 1]; k < L[j] - 1; k++) {
					T0[k * mid1 + j] = T0[k * mid1 + j + 1];
				}
				T0[(L[j] - 1) * mid1 + j] = T0[(L[j] - 2) * mid1 + j] - step * (c * r0) / (10000* l * lambda ) * jem(Table, r0, l * T[(L[j] - 1) * size1 + j] / c, E2[(L[j] - 1) * mid1 + j]) * E(T[(L[j] - 1) * size1 + j]);
				//std::cout << step * (c * r0) / (10000* l  * lambda ) * jem(Table, r0, l * T[(L[j] - 1) * size1 + j] / c, E2[(L[j] - 1) * mid1 + j]) * E(T[(L[j] - 1) * size1 + j]) << std::endl;
			}
			else
				T0[(L[j] - 1) * mid1 + j] = T0[(L[j] - 2) * mid1 + j] - step * (c * r0) / (10000* l * lambda) * jem(Table,r0,l*T[(L[j] - 1) * size1 + j]/c,E2[(L[j] - 1) * mid1 + j])*E(T[(L[j] - 1) * size1 + j]); 
		}

		//for (int j = 1; j < mid2; j++) {									//под номером четыре вторая точка внутри катода, до середины ( не включая)
		//	for (int i = 1; i < size2; i++) {									//Обход внутренних точек катода
		//		T0[i * mid1 + j] =
		//			((T[i * size1 + j - 1] - T[i * size1 + j]) * ar[j - 1] - (T[i * size1 + j] - T[i * size1 + j + 1]) * ar[j]) * YT /xcent[j] * koeflaplas +
		//			((T[(i + 1) * size1 + j] - T[i * size1 + j]) - (T[i * size1 + j] - T[(i - 1) * size1 + j])) * YT * koeflaplas + T[i * size1 + j];
		//		//+ s2 * pow(E1[j * size1 + i],2) * (tau * t0) / (l * rho);
		//	}
		//}
		//for (int i = 1; i < size2; i++)
		//	T0[i * mid1 + mid2] = T0[i * mid1 + mid2 - 1];
		//for (int j = 0; j < (size1 + 1) / 2; j++)
		//	T0[j] = T0[(size1 + 1) / 2 + j];					//Условие на нижней границе - производная равна 0.

		//for (int j = 0; j < size2; j++)
		//	T0[j*mid1] = T0[j*mid1 + 1];

		//for (int j = 0; j < mid1; j++)
		//	T0[(size2 - 1) * mid1 + j] = T0[(size2 - 2) * mid1 + j] - 1;

		//output_vtk_binary(T, size1 - 1, size2 - 1, x, y, get_name(DirName, "densT", count));
		for (int i = 0; i < size2; i++) {
			for (int j = 0; j < mid1; j++) {
				T[i * size1 + j] = T0[i * mid1 + j];
				T[(i + 1) * size1 - 1 - j] = T0[i * mid1 + j];
				if (count % 40000 == 0) {
					buf1 = T[i * size1 + j] - T1[i * mid1 + j];
					if (buf < buf1)
						buf = buf1;
					T1[i * mid1 + j] = T0[i * mid1 + j];
				}
			}
		}
		//output_vtk_binary(t, size1 - 1, size2 - 1, x, y, get_name(DirName, "densT", count));
		if (count % 40000 == 0) {
			dif = buf;
			std::cout <<count<<" " << dif << std::endl;
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
	std::cout << count << std::endl;
	int end = time(NULL);
	std::cout << "Время работы: " << (end - sta) << std::endl;
	//output_vtk_binary(T, size1 - 1, size2 - 1, x, y, get_name(DirName, "densT", count)); 
	delete[] L;
	delete[] T0;
	delete[] T1;
	delete[] u0;
	delete[] u1;
	delete[] delt;
	delete[] ax;
	delete[] ay;
	delete[] ar;
	delete[] E1;
	delete[] E2;
	delete[] E3;
	delete[] E4;
	delete[] E5;
}

void Tablereader(double** Table) {
		std::string line;
		//std::ifstream in("D:\\ProjectM2\\Table1.txt"); // окрываем файл для чтения
		std::ifstream in("Table1.txt"); // окрываем файл для чтения
		if (in.is_open())
		{
			int count1 = 0;
			while (getline(in, line))
			{
				int count = 0;
				int i = 0, j = 0;
				while (i <= line.find_last_of(" "))
				{
					i = line.find(" ", i + 1);
					Table[count1][count] = std::stod(line.substr(j, i - j));
					count++;
					j = i + 1;
				}
				count1++;
			}
		}
		in.close();     // закрываем файл
}

double jem(double **Table, double r0,double T, double E) {
	double ee,tt,j = 0;
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
	double x0, y0;
	x0 = (tt - 300) / 50;
	y0 = (ee - 10) / 9999.99;
	k[0] = floor(x0);
	k[1] = floor(y0);
	k[2] = k[0] + 1;
	k[3] = k[1] + 1;
	j = Table[k[0]][k[1]] * (k[2] - x0) * (k[3] - y0) + Table[k[2]][k[1]] * (x0 - k[0]) * (k[3] - y0) +
		Table[k[0]][k[3]] * (k[2] - x0) * (y0 - k[1]) + Table[k[2]][k[3]] * (x0 - k[0]) * (y0 - k[1]);
	//std::cout << tt << " " << E <<" "<<r0<<" " << " " << j << std::endl;
	return(j);
}

double Ef(double* E1, double* E2, int mid1, int size2,int sir, int siv, int i, int j) {
	double cos, sin, x, y;
	x = abs(mid1 - 1 - j);
	y = abs(size2 - sir - siv - 1 - i);
	cos = x / sqrt(pow(x, 2) + pow(y, 2));
	sin = y / sqrt(pow(x, 2) + pow(y, 2));
	return((pow(E1[i * mid1 + j]*cos, 2) + pow(E2[i * mid1 + j]*sin, 2)));
}

double E(double T) {
	double E;
	if (T < 0.2129061)
		E = -0.0589529 * T / tan(14.6137 * T);
	else
		E = 4.42871 + 0.0417038 * T + (-21, 8518 + 0.25058 * T) / (4.92306 + pow((-1 + 9.30338 * T), 3.48481));
	return (E);
}

double Dcoef(double **Table, double* u, double *T, int* L,int size1, int step , int s2, double r0) {
	double D = 0;
	for (int j = (size1 - 1) / 2 - 5; j < (size1 - 1) / 2; j++) {
		for (int i = L[j - 1] - 1; i < L[j]; i++) {
			D += jem(Table, r0, T[i * size1 + j], sqrt(pow(u[(i + 1) * size1 + j] - u[i * size1 + j], 2) + pow(u[i * size1 + j] - u[i * size1 + j + 1], 2)) / step) / (s2 * sqrt(pow(u[i * size1 + j] - u[(i - 1) * size1 + j], 2) + pow(u[i * size1 + j] - u[i * size1 + j + 1], 2)) / step);
		}
	}
	D += D;
	int j = (size1 - 1) / 2;
	int i = L[j] - 1;
	D += jem(Table, r0, T[i * size1 + j], sqrt(pow(u[(i + 1) * size1 + j] - u[i * size1 + j], 2) + pow(u[i * size1 + j] - u[i * size1 + j + 1], 2)) / step) / (s2 * sqrt(pow(u[i * size1 + j] - u[(i - 1) * size1 + j], 2) + pow(u[i * size1 + j] - u[i * size1 + j + 1], 2)) / step);
	D = D / 11;
	return D;
}

void derivative(double* u, int size1, int size2, double step, std::string DirName)
{
	double der;
	int mid;
	mid = (size1 + 1) / 2;
	for (int i = 0; i < size2 - 1; i++)
	{
		der = (u[(i + 1) * size1 + mid] - u[i * size1 + mid]) / step;
		std::ofstream out(get_name(DirName, "derivative"), std::ios::app);
		if (out.is_open())
		{
			out << i*step << "\t"<<der<< std::endl;
		}
		out.close();
	}
}

std::string get_dir_name()
{
	std::string DirTimeName;
	/*time_t rawtime;
	struct tm timeinfo;
	time(&rawtime);
	localtime_s(&timeinfo ,&rawtime);
	DirTimeName = std::to_string((&timeinfo)->tm_hour) + "_" + std::to_string((&timeinfo)->tm_min) + "___" + std::to_string((&timeinfo)->tm_mday) + "_" + std::to_string((&timeinfo)->tm_mon + 1);*/
	DirTimeName = "100";
	return DirTimeName;
}

std::string get_name(std::string name, int k)
{
	std::string st2;
	st2 = name;
	std::ostringstream s;
	s << k;
	st2 += s.str();
	st2 += ".vtk";
	return (st2);
}

std::string get_name(std::string DirName, std::string name)
{
	std::string st2;
	st2 = name;
	st2 += ".txt";
	return (DirName + "\\" + st2);
}

std::string get_name(std::string DirName, std::string name, int k)
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
void SwapEnd(T& var)
{
	char* varArray = reinterpret_cast<char*>(&var);
	for (long i = 0; i < static_cast<long>(sizeof(var) / 2); i++)
		std::swap(varArray[sizeof(var) - 1 - i], varArray[i]);
}

void output_vtk_binary(double* u, int n1, int n2, double* x, double* y, std::string name)
{

	int arr_size = (n1 + 1) * (n2 + 1);
	double* my_u = new double[arr_size];
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
	fileD << "POINTS " << arr_size << " double" << std::endl;
	double tt1, tt2, tt3 = 0;
	SwapEnd(tt3);
	for (int j = 0; j <= n2; j++)
	{
		for (int i = 0; i <= n1; i++)
		{
			tt1 = x[i];
			tt2 = y[j];
			SwapEnd(tt1);
			SwapEnd(tt2);

			fileD.write((char*)&tt1, sizeof(double));
			fileD.write((char*)&tt2, sizeof(double));
			fileD.write((char*)&tt3, sizeof(double));

		}
	}
	fileD << "POINT_DATA " << arr_size << std::endl;
	fileD << "SCALARS phi double 1" << std::endl;
	fileD << "LOOKUP_TABLE default" << std::endl;

	for (int j = 0; j <= n2; j++)
	{
		for (int i = 0; i <= n1; i++)
		{
			fileD.write((char*)&my_u[j * (n1 + 1) + i], sizeof(double));
		}
	}


	fileD.close();


	delete[] my_u;
}