#include <iostream>
#include <memory>
#include <omp.h>
#include <thread>
#include <mutex>
#include <vector>
#include <chrono>
#include "Header.h"
#include <fstream>
#include <queue>


double average(const double* v, size_t n) {
	double res = 0.0;
	for (size_t i = 0; i < n; i++) res += v[i];
	return res / n;
}

double average_reduce(const double* v, size_t n) {
	double res = 0.0;
#pragma omp parallel for reduction(+: res)
	for (int i = 0; i < n; i++) res += v[i];
	return res / n;
}

double average_rr(const double* v, size_t n) {
	double res = 0.0;
#pragma omp parallel
	{
		unsigned t = omp_get_thread_num();
		unsigned T = omp_get_max_threads();
		for (int i = t; i < n; i += T) res += v[i];
	}
	return res / n;
}

double average_omp(const double* v, size_t n) {
	double res = 0.0, * partial_sums = (double*)calloc(omp_get_num_procs(), sizeof(double));
#pragma omp parallel
	{
		unsigned t = omp_get_thread_num();
		unsigned T = omp_get_max_threads();
		for (int i = t; i < n; i += T) partial_sums[t] += v[i];
	}
	for (size_t i = 1; i < omp_get_num_procs(); ++i)
		partial_sums[0] += partial_sums[i];
	res = partial_sums[0] / n;
	free(partial_sums);
	return res;
}

double average_omp_modified(const double* v, size_t n) {
	unsigned T;
	double res = 0.0;
	partial_sum_t2* partial_sums;
#pragma omp parallel
	{
		unsigned t = omp_get_thread_num();

#pragma omp single
		{
			T = omp_get_max_threads();
			partial_sums = (partial_sum_t2*)malloc(T * sizeof partial_sum_t2);
		}
		partial_sums[t].value = 0.0;
		for (int i = t; i < n; i += T) partial_sums[t].value += v[i];
	}
	for (size_t i = 1; i < omp_get_num_procs(); ++i)
		partial_sums[0].value += partial_sums[i].value;
	res = partial_sums[0].value / n;
	free(partial_sums);
	return res;
}

double average_omp_mtx(const double* v, size_t n) {
	double res = 0.0;
#pragma omp parallel
	{
		unsigned T = omp_get_max_threads();
		unsigned t = omp_get_thread_num();
		for (int i = t; i < n; i += T) {
#pragma omp critical
			{
				res += v[i];
			}
		}
	}
	return res / n;
}

double average_omp_mtx_modified(const double* v, size_t n) {
	double res = 0.0;
	size_t T, t;
#pragma omp parallel
	{
		double partial_sum = 0.0;
		T = omp_get_max_threads();
		t = omp_get_thread_num();
		for (size_t i = t; i < n; i += T)
			partial_sum += v[i];
#pragma omp critical
		{
			res += partial_sum;
		}
	}
	return res / n;
}

double average_cpp_mtx(const double* v, size_t n) {
	unsigned T = omp_get_max_threads();
	double res = 0.0;
	std::vector<std::thread> workers;
	std::mutex mtx;
	auto worker_proc = [&mtx, T, v, n, &res](unsigned t) {
		double partial_result = 0.0;
		for (std::size_t i = t; i < n; i += T)
			partial_result += v[i];
		mtx.lock();
		res += partial_result;
		mtx.unlock();
		};
	for (unsigned t = 0; t < T; ++t)
		workers.emplace_back(worker_proc, t);
	for (auto& w : workers)
		w.join();
	return res / n;
}

double average_cpp_mtx_modified0(const double* v, size_t n) {
	unsigned T = omp_get_max_threads();
	double res = 0.0;
	std::vector<std::thread> workers;
	std::mutex mtx;
	auto worker_proc = [&mtx, &res, T, v, n](unsigned t) {
		double partial_result = 0.0;
		for (std::size_t i = t; i < n; i += T)
			partial_result += v[i];
		mtx.lock();
		res += partial_result;
		mtx.unlock();
		};
		for (unsigned int t = 1; t < T; ++t)
			workers.emplace_back(worker_proc, t);
		worker_proc(0);
		for (auto& w : workers)
			w.join();
		return res / n;
}

double average_cpp_mtx_modified1(const double* v, size_t n) {
	size_t T = omp_get_max_threads();
	double res = 0.0;
	std::vector<std::thread> workers;
	std::mutex mtx;
	auto worker_proc = [&mtx, T, v, n, &res](size_t t) {
		double partial_result = 0.0;
		for (std::size_t i = t; i < n; i += T)
			partial_result += v[i];
		std::scoped_lock l{ mtx };
		res += partial_result;
		};
	for (unsigned int t = 1; t < T; ++t) {
		workers.emplace_back(worker_proc, t);
	}
	worker_proc(0);
	for (auto& w : workers)
		w.join();
	return res / n;
}

double average_omp_aligned(const double* V, size_t n) {
	unsigned T;
	double res = 0.0;
	partial_sum_t1* partial_sums;

#pragma omp parallel shared(T)
	{
		unsigned t = omp_get_thread_num();
		T = omp_get_num_threads();
#pragma omp single
		{
			partial_sums = (partial_sum_t1*)malloc(T * sizeof(partial_sum_t1));
		}
		partial_sums[t].value = 0.0;
		for (int i = t; i < n; i += T) {
			partial_sums[t].value += V[i];
		}
	}
	for (size_t i = 1; i < T; ++i) {
		partial_sums[0].value += partial_sums[i].value;
	}
	res = partial_sums[0].value / n;
	free(partial_sums);
	return res;
}

double average_cpp_aligned(const double* V, size_t n) {
	unsigned T;
	double res = 0.0;
	std::unique_ptr<partial_sum_t1[]> partial_sums;
#pragma omp parallel shared(T)
	{
		unsigned t = omp_get_thread_num();
		T = omp_get_num_threads();
#pragma omp single
		{
			partial_sums = std::make_unique<partial_sum_t1[]>(T);
		}

		partial_sums[t].value = 0.0;
		for (int i = t; i < n; i += T) {
			partial_sums[t].value += V[i];
		}
	}
	for (size_t i = 1; i < T; ++i) {
		partial_sums[0].value += partial_sums[i].value;
	}
	return partial_sums[0].value / n;
}

double average_cpp_mtx_local(const double* v, size_t N) {
	size_t T = omp_get_max_threads();
	double average = 0.0;
	std::mutex mtx;
	std::vector<std::thread> threads;

	auto worker = [&mtx, T, N, v, &average](size_t t) {
		size_t e = N / T;
		size_t b = N % T;
		if (t < b)
			b = t * ++e;
		else
			b += t * e;
		e += b;
		double partial_sum = 0.0;
		for (size_t i = b; i < e; i++)
			partial_sum += v[i];
		std::scoped_lock l{ mtx };
		average += partial_sum;
		};

	for (unsigned int t = 1; t < T; ++t) {
		threads.emplace_back(worker, t);
	}

	worker(0);
	for (auto& w : threads)
		w.join();
	return average / N;

}

double get_omp_time(double (*f)(const double*, size_t), const double* v, size_t n) {
	auto t1 = omp_get_wtime();
	f(v, n);
	return omp_get_wtime() - t1;
}

template <std::invocable<const double*, size_t> F>
auto cpp_get_time(F f, const double* V, size_t n) {
	using namespace std::chrono;
	auto t1 = steady_clock::now();
	f(V, n);
	auto t2 = steady_clock::now();
	return duration_cast<milliseconds>(t2 - t1).count();
}

template <typename F>
auto run_experiment(F f, const double* v, std::size_t n)
	requires std::is_invocable_r_v<double, F, const double*, std::size_t> {
	std::vector<profiling_results_t> r;
	std::size_t T_max = get_num_threads();
	for (std::size_t T = 1; T <= T_max; ++T) {
		set_num_threads(T);
		using namespace std::chrono;
		auto t0 = std::chrono::steady_clock::now();
		auto rr_result = f(v, n);
		auto t1 = std::chrono::steady_clock::now();
		profiling_results_t rr;
		r.push_back(rr);
		unsigned int times = duration_cast<nanoseconds> (t1 - t0).count();
		r[T - 1].time = times;
		r[T - 1].result = rr_result;
		r[T - 1].T = T;
		r[T - 1].speedup = r[0].time / r[T - 1].time;
		r[T - 1].efficiency = r[T - 1].speedup / T;
	}
	return r;
}

double average_cpp_reduction(const double* v, size_t N) {
	size_t T = omp_get_max_threads();
	double average = 0.0;
	std::vector<std::thread> threads;
	double* partitial_results = new double[T];
	for (int i = 0; i < T; ++i)
		partitial_results[i] = 0;
	barrier bar = barrier(T);

	auto worker = [&partitial_results, &bar, T, N, v](size_t t) {
		size_t e = N / T;
		size_t b = N % T;
		if (t < b)
			b = t * ++e;
		else
			b += t * e;
		e += b;

		double partial_sum = 0.0;
		for (size_t i = b; i < e; i++)
			partial_sum += v[i];
		partitial_results[t] = partial_sum;

		for (size_t step = 1, next = 2; step < T; step = next, next += next) {
			bar.arrive_and_wait();
			if (((t & (next - 1)) == 0) && t + step < T) {
				partitial_results[t] += partitial_results[t + step];
			}
		}
	};

	for (unsigned int t = 1; t < T; ++t) {
		threads.emplace_back(worker, t);
	}

	worker(0);
	for (auto& w : threads)
		w.join();
	return partitial_results[0]/N;

}


int main() {
	std::size_t N = 1u << 25;
	auto buf = std::make_unique<double[]>(N);
	for (std::size_t i = 0; i < N; ++i) buf[i] = i;


	

	/*auto t1 = omp_get_wtime();
	auto v1 = average(buf.get(), N);
	auto t2 = omp_get_wtime();
	std::cout << "Result (classical): " << v1 << "\n";
	std::cout << "Time taken (classical): " << t2 - t1 << "\n";
	auto res = run_experiment(average, buf.get(), N);
	std::cout << "Result,Time,Speedup,Efficiency(average)" << "\n";
	for (int i = 0; i < res.size(); i++) {
		std::cout << res[i].result << "," << res[i].time << "," << res[i].speedup << "," << res[i].efficiency << "\n";
	}*/

	/*auto t3 = omp_get_wtime();
	auto v2 = average_reduce(buf.get(), N);
	auto t4 = omp_get_wtime();
	std::cout << "Result (reduce): " << v2 << "\n";
	std::cout << "Time taken (reduce): " << t4 - t3 << "\n";
	auto res = run_experiment(average_reduce, buf.get(), N);
	std::cout << "Result,Time,Speedup,Efficiency(average_reduce)" << "\n";
	for (int i = 0; i < res.size(); i++) {
		std::cout << res[i].result << "," << res[i].time << "," << res[i].speedup << "," << res[i].efficiency << "\n";
	}*/

	/*auto t5 = omp_get_wtime();
	auto v3 = average_rr(buf.get(), N);
	auto t6 = omp_get_wtime();
	std::cout << "Result (Round Robin): " << v3 << "\n";
	std::cout << "Time taken (Round Robin): " << t6 - t5 << "\n";

	res = run_experiment(average_rr, buf.get(), N);
	std::cout << "Result,Time,Speedup,Efficiency(average_rr)" << "\n";
	for (int i = 0; i < res.size(); i++) {
		std::cout << res[i].result << "," << res[i].time << "," << res[i].speedup << "," << res[i].efficiency << "\n";
	}*/


	/*auto t7 = omp_get_wtime();
	auto v4 = average_omp(buf.get(), N);
	auto t8 = omp_get_wtime();
	std::cout << "Result (Open MP): " << v4 << "\n";
	std::cout << "Time taken (Open MP): " << t8 - t7 << "\n";
	auto res = run_experiment(average_omp, buf.get(), N);
	std::cout << "Result,Time,Speedup,Efficiency(average_omp)" << "\n";
	for (int i = 0; i < res.size(); i++) {
		std::cout << res[i].result << "," << res[i].time << "," << res[i].speedup << "," << res[i].efficiency << "\n";
	}*/

	/*auto t9 = omp_get_wtime();
	auto v5 = average_omp_modified(buf.get(), N);
	auto t10 = omp_get_wtime();
	std::cout << "Result (Open MP modified): " << v5 << "\n";
	std::cout << "Time taken (Open MP modified): " << t10 - t9 << "\n";
	auto res = run_experiment(average_omp_modified, buf.get(), N);
	std::cout << "Result,Time,Speedup,Efficiency(average_omp_modified)" << "\n";
	for (int i = 0; i < res.size(); i++) {
		std::cout << res[i].result << "," << res[i].time << "," << res[i].speedup << "," << res[i].efficiency << "\n";
	}*/

	auto t11 = omp_get_wtime();
	auto v6 = average_omp_mtx(buf.get(), N);
	auto t12 = omp_get_wtime();
	std::cout << "Result (Open MP Mutex): " << v6 << "\n";
	std::cout << "Time taken (Open MP Mutex): " << t12 - t11 << "\n";
	auto res = run_experiment(average_omp_mtx, buf.get(), N);
	std::cout << "Result,Time,Speedup,Efficiency(average_omp_mtx)" << "\n";
	for (int i = 0; i < res.size(); i++) {
		std::cout << res[i].result << "," << res[i].time << "," << res[i].speedup << "," << res[i].efficiency << "\n";
	}

	/*auto t13 = omp_get_wtime();
	auto v7 = average_omp_mtx_modified(buf.get(), N);
	auto t14 = omp_get_wtime();
	std::cout << "Result (Open MP Mutex modified): " << v7 << "\n";
	std::cout << "Time taken (Open MP Mutex modified): " << t14 - t13 << "\n";

	res = run_experiment(average_omp_mtx_modified, buf.get(), N);
	std::cout << "Result,Time,Speedup,Efficiency(average_omp_mtx_modified)" << "\n";
	for (int i = 0; i < res.size(); i++) {
		std::cout << res[i].result << "," << res[i].time << "," << res[i].speedup << "," << res[i].efficiency << "\n";
	}*/

	/*auto t15 = omp_get_wtime();
	auto v8 = average_cpp_mtx(buf.get(), N);
	auto t16 = omp_get_wtime();
	std::cout << "Result (CPP Mutex): " << v8 << "\n";
	std::cout << "Time taken (CPP Mutex): " << t16 - t15 << "\n";

	res = run_experiment(average_cpp_mtx, buf.get(), N);
	std::cout << "Result,Time,Speedup,Efficiency(average_cpp_mtx)" << "\n";
	for (int i = 0; i < res.size(); i++) {
		std::cout << res[i].result << "," << res[i].time << "," << res[i].speedup << "," << res[i].efficiency << "\n";
	}*/

	/*auto t17 = omp_get_wtime();
	auto v9 = average_cpp_mtx_modified1(buf.get(), N);
	auto t18 = omp_get_wtime();
	std::cout << "Result (CPP Mutex modified0): " << v9 << "\n";
	std::cout << "Time taken (CPP Mutex modified0): " << t18 - t17 << "\n";

	res = run_experiment(average_cpp_mtx_modified1, buf.get(), N);
	std::cout << "Result,Time,Speedup,Efficiency(average_cpp_mtx_modified1)" << "\n";
	for (int i = 0; i < res.size(); i++) {
		std::cout << res[i].result << "," << res[i].time << "," << res[i].speedup << "," << res[i].efficiency << "\n";
	}*/

	/*auto t19 = omp_get_wtime();
	auto v10 = average_cpp_mtx_modified1(buf.get(), N);
	auto t20 = omp_get_wtime();
	std::cout << "Result (CPP Mutex modified1): " << v10 << "\n";
	std::cout << "Time taken (CPP Mutex modified1): " << t20 - t19 << "\n";

	res = run_experiment(average_cpp_mtx_modified1, buf.get(), N);
	std::cout << "Result,Time,Speedup,Efficiency(average_cpp_mtx_modified1)" << "\n";
	for (int i = 0; i < res.size(); i++) {
		std::cout << res[i].result << "," << res[i].time << "," << res[i].speedup << "," << res[i].efficiency << "\n";
	}*/

	/*auto t21 = omp_get_wtime();
	auto v11 = average_cpp_mtx_local(buf.get(), N);
	auto t22 = omp_get_wtime();
	std::cout << "Result (CPP Mutex local): " << v11 << "\n";
	std::cout << "Time taken (CPP Mutex local): " << t22 - t21 << "\n";

	res = run_experiment(average_cpp_mtx_local, buf.get(), N);
	std::cout << "Result,Time,Speedup,Efficiency(average_cpp_mtx_local)" << "\n";
	for (int i = 0; i < res.size(); i++) {
		std::cout << res[i].result << "," << res[i].time << "," << res[i].speedup << "," << res[i].efficiency << "\n";
	}*/

	/*auto t23 = omp_get_wtime();
	auto v12 = average_cpp_reduction(buf.get(), N);
	auto t24 = omp_get_wtime();
	std::cout << "Result (CPP reduction): " << v12 << "\n";
	std::cout << "Time taken (CPP redaction): " << t24 - t23 << "\n";

	res = run_experiment(average_cpp_reduction, buf.get(), N);
	std::cout << "Result,Time,Speedup,Efficiency(average_cpp_reduction)" << "\n";
	for (int i = 0; i < res.size(); i++) {
		std::cout << res[i].result << "," << res[i].time << "," << res[i].speedup << "," << res[i].efficiency << "\n";
	}*/


	/*auto t25 = omp_get_wtime();
	auto v13 = average_omp_aligned(buf.get(), N);
	auto t26 = omp_get_wtime();
	std::cout << "Result (OMP aligned): " << v12 << "\n";
	std::cout << "Time taken (OMP aligned): " << t24 - t23 << "\n";

	res = run_experiment(average_omp_aligned, buf.get(), N);
	std::cout << "Result,Time,Speedup,Efficiency(average_omp_aligned)" << "\n";
	for (int i = 0; i < res.size(); i++) {
		std::cout << res[i].result << "," << res[i].time << "," << res[i].speedup << "," << res[i].efficiency << "\n";
	}*/


	/*auto t27 = omp_get_wtime();
	auto v14 = average_cpp_aligned(buf.get(), N);
	auto t28 = omp_get_wtime();
	std::cout << "Result (cpp aligned): " << v12 << "\n";
	std::cout << "Time taken (cpp aligned): " << t24 - t23 << "\n";

	res = run_experiment(average_cpp_aligned, buf.get(), N);
	std::cout << "Result,Time,Speedup,Efficiency(average_cpp_aligned)" << "\n";
	for (int i = 0; i < res.size(); i++) {
		std::cout << res[i].result << "," << res[i].time << "," << res[i].speedup << "," << res[i].efficiency << "\n";
	}*/

	return 0;
}