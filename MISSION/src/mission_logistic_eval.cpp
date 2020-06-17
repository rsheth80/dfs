#include "MurmurHash.h"
#include "fast_parser.h"
#include "mp_queue.h"
#include "cms.h"
#include "topk.h"

#include <stdlib.h>
#include <vector>
#include <utility>
#include <iostream>
#include <climits>
#include <random>
#include <chrono>

#include <thread>
#include <mutex>
#include <omp.h>

// modded RS
#include <fstream>
#include <unordered_set>
#include <unordered_map>
// modded RS

/***** Hyper-Parameters *****/

// Size of Top-K Heap
const size_t TOPK = (1 << 14) - 1;

// Number of Classes - Always 1 for Logistic Regression
const size_t K = 1;

// Size of Count-Sketch Array
const size_t D = (1 << 18) - 1;

// Number of Arrays in Count-Sketch
const size_t N = 3;

// Learning Rate
const float LR = 5e-1;

/***** End of Hyper-Parameters *****/

typedef std::pair<int, float> fp_t;
typedef std::pair<int, std::vector<fp_t> > x_t;
typedef TopK<int, TOPK> tk_t;

// Serialize Output
std::mutex mtx;

// modded RS
bool neglabel;
std::unordered_set<size_t> f_ix;
std::unordered_map<size_t, float> beta;
/*
// Number of threads for parallel data preprocessing
const size_t THREADS = 2;
// Maximum number of features for an example
const size_t MAX_FEATURES = 5000;

std::array<std::array<hc<N>, MAX_FEATURES>, THREADS> caches;
*/

void split(data_t& item, fp_t& result)
{
		data_t key;
		data_t value;
		memset(key.data(), 0, key.size());
		memset(value.data(), 0, value.size());

		int cdx = 0;
		for(char v = item[cdx]; v != ':'; v = item[cdx])
		{
				key[cdx++] = v;
		}
		result.first = atoi(key.data());

// modded RS
        if(f_ix.find(result.first) == f_ix.end())
            result.second = 0.0f;

        else
        {
            int initial = ++cdx;
            for(char v = item[cdx]; v != 0; v = item[cdx])
            {
                    value[(cdx++ - initial)] = v;
            }
            result.second = atof(value.data());
        }
//        int initial = ++cdx;
//        for(char v = item[cdx]; v != 0; v = item[cdx])
//        {
//                value[(cdx++ - initial)] = v;
//        }
//        result.second = atof(value.data());
// modded RS
}

void producer(fast_parser& p, mp_queue<x_t>& q)
{
		for(std::vector<data_t> x = p.read(' '); p; x = p.read(' '))
		{
				const int label = atoi(x[0].data());

				// Parse Features
				std::vector<fp_t> features(x.size()-1);
				for(size_t idx = 1; idx < x.size(); ++idx)
				{
                    split(x[idx], features[idx-1]);
				}

				q.enqueue(std::make_pair(label, features));
		}
		//std::cout << "Finished Reading" << std::endl;
}

// modded RS
//float process(CMS<N>& sketch, tk_t& topk, const x_t& x, bool train)
float process(const x_t& x, bool train)
// modded RS
{
// modded RS
        float label;
        if(neglabel)
		    label = (x.first + 1.0f)/2.0f;
        else
            label = x.first + 0.0f;
//		float label = (x.first + 1.0f)/2.0f;
// modded RS
		const std::vector<fp_t>& features = x.second;

// modded RS
        /*
		const int tid = omp_get_thread_num();
		std::array<hc<N>, MAX_FEATURES>& cache = caches[tid];
		for(size_t idx = 0; idx < features.size(); ++idx)
		{
				const void * key_ptr = (const void *) &features[idx].first;
				sketch.hash(key_ptr, sizeof(int), cache[idx]);
		}
        */
// modded RS

		float logit = 0;
// modded RS
        /*
		for(size_t idx = 0; idx < features.size(); ++idx)
  	    {
				logit += topk[item.first] * item.second;
		}
        */
		for(const fp_t& item : features)
        {
            logit += item.second*beta[item.first];
		}
// modded RS

		float sigmoid = 1.0 / (1.0 + std::exp(-logit));
		float loss = (label * std::log(sigmoid) + (1.0 - label) * std::log(1 - sigmoid));
		if(!train)
		{
				mtx.lock();
				std::cout << label << " " << sigmoid << std::endl;
				mtx.unlock();
				return loss;
		}

		float gradient = label - sigmoid;
		for(size_t idx = 0; idx < features.size(); ++idx)
		{
// modded RS
            /*
				float value = sketch.update(cache[idx], LR * gradient * features[idx].second);
				topk.push(features[idx].first, value);
            */
            if(f_ix.find(features[idx].first) != f_ix.end())
                beta[features[idx].first] += features[idx].second*LR*gradient;
// modded RS
		}

		return loss;
}

void consumer(fast_parser& p, mp_queue<x_t>& q, bool train)
{
		std::vector<x_t> items;
		size_t cnt = 0;
		while(p || q)
		{
				if(!q.full() && p)
				{
						std::this_thread::sleep_for (std::chrono::seconds(1));
						continue;
				}

				// Retrieve items from multiprocess queue
				q.retrieve(items);
				cnt += items.size();
				float loss = 0.0;
				for(size_t cdx = 0; cdx < items.size(); ++cdx)
				{
// modded RS
//                      loss += process(sketch, topk, items[cdx], train);
						loss += process(items[cdx], train);
// modded RS
				}

				// Debug
// modded RS
//				if(train)
				if(0*train)
// modded RS
				{
						float avg_loss = -loss / items.size();
						std::cout << cnt << " " << avg_loss << std::endl;
				}
				items.clear();
		}
		//std::cout << "Finished Consumer" << std::endl;
}

int main(int argc, char* argv[])
{
// modded RS
		//CMS<N> sketch(K, D);
// modded RS
		mp_queue<x_t> q(10000);
// modded RS
		//tk_t topk;

        std::ifstream f(argv[3]);
        if(!f.good())
        {
            std::cout << "Feature indices file does not exist." << std::endl;
            return 1;
        }
        while(!f.eof())
        {
            size_t ix;
            f >> ix;
            f_ix.insert(ix);
            beta.insert(std::make_pair(ix, 0.0f));
        }
        neglabel = atoi(argv[4]);
// modded RS

		fast_parser train_p(argv[1]);
		std::thread train_pr([&] { producer(train_p, q); });
// modded RS
//		std::thread train_cr([&] { consumer(sketch, topk, train_p, q, true); });
		std::thread train_cr([&] { consumer(train_p, q, true); });
// modded RS
		train_pr.join();
		train_cr.join();

		fast_parser test_p(argv[2]);
		std::thread test_pr([&] { producer(test_p, q); });
// modded RS
//		std::thread test_cr([&] { consumer(sketch, topk, test_p, q, false); });
		std::thread test_cr([&] { consumer(test_p, q, false); });
// modded RS
		test_pr.join();
		test_cr.join();

		return 0;
}
