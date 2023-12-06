#pragma once
#include <thread>
#include <omp.h>

static unsigned g_thread_num = std::thread::hardware_concurrency();


unsigned get_num_threads() {
	return g_thread_num;
}

void set_num_threads(unsigned T) {
	g_thread_num = T;
	omp_set_num_threads(T);
}

class latch {
	unsigned T;
	std::mutex mtx;
	std::condition_variable cv;
public:
	latch(unsigned threads) : T(threads) {}
	void arrive_and_wait() {
		std::unique_lock l{ mtx };
		if (--T) do {
			cv.wait(l);
		} while (T > 0);
		else
			cv.notify_all();
	}
};

class barrier {
	unsigned lock_id = 0;
	unsigned T, Tmax;
	std::mutex mtx;
	std::condition_variable cv;
public:
	barrier(unsigned threads) : T(threads), Tmax(threads) {}
	void arrive_and_wait() {
		std::unique_lock l{ mtx };
		if (--T) {
			unsigned my_lock_id = lock_id;
			while (my_lock_id == lock_id)
				cv.wait(l);
		}
		else {
			++lock_id;
			T = Tmax;
			cv.notify_all();
		}
	}
};

struct profiling_results_t {
	double result;
	double time;
	double speedup, efficiency;
	unsigned T;
};

struct partial_sum_t1 {
	alignas(64) double value;
};

struct partial_sum_t2 {
	union {
		double value;
		char padd[64];
	};
};