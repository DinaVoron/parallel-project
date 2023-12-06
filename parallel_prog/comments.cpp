//class barrier {
//	unsigned lock_id = 0;
//	unsigned T, Tmax;
//	std::mutex mtx;
//	std::condition_variable cv;
//public:
//	barrier(unsigned threads) : T(threads), Tmax(threads) {}
//	void arrive_and_wait() {
//		std::unique_lock l{ mtx };
//		if (--T) do {
//			cv.wait(l);
//		} while (T > 0 && T < Tmax);
//		else {
//			T = Tmax;
//			cv.notify_all();
//		}
//	}
//};

//double average_cpp_reduction(const double* v, size_t N, barrier bar) {
//	size_t T = std::thread::hardware_concurrency();
//	double partial_results[T];
//	for (size_t step = 1, next = 2; step < T, step = next; next += next) {
//		bar.arrive_and_wait();
//		if ((t & (next) == 0) && t + step < T) {
//			partial_results[t] += partial_results[t + step];
//		}
//	}
//	return partial_results[0]/N;
//}


//auto run_experiment(double (*f)(const double*, size_t), const double* v, size_t n) {
//	struct profiling_results_t* res_table = (profiling_results_t*)malloc(omp_get_num_procs() * sizeof(struct profiling_results_t));
//	for (unsigned T = 1; T <= omp_get_num_procs(); T++) {
//		omp_set_num_threads(T);
//		auto t1 = omp_get_wtime();
//		res_table[T - 1].result = f(v, n);
//		auto t2 = omp_get_wtime();
//		res_table[T - 1].speedup = res_table[0].time / (t2 - t1);
//		res_table[T - 1].efficiency = res_table[T - 1].speedup / T;
//		res_table[T - 1].T = T;
//	}
//}

/*std::condition_variable cv;
	std::queue<int> q;
	std::mutex mtx;
	unsigned P = std::thread::hardware_concurrency();
	auto producers = 1;
	auto consumers = P - producers;
	std::vector<std::thread> consumer_v;
	std::vector<std::thread> producer_v;
	for (unsigned i = 0; i < producers; i++) {
		producer_v.emplace_back([&q, &mtx, &consumers, &cv]() {
			for (unsigned c = 0; c < consumers; c++) {
				std::scoped_lock lock(mtx);
				q.push(c);
			}
			cv.notify_one();
		});
	}

	for (unsigned i = 0; i < consumers; i++) {
		consumer_v.emplace_back([&q, &mtx, &cv](unsigned t) {
			std::unique_lock ul(mtx);
			while (q.empty()) {
				cv.wait(ul);
			}
			int m = q.front();
			q.pop();

			std::cout << "thread " << t << " recieved message " << m << "\n";
			ul.unlock();
		}, i);
	}

	for (auto& producer : producer_v)
		producer.join();

	for (auto& consumer : consumer_v)
		consumer.join();*/