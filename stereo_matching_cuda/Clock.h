#pragma once

#include "SystemIncludes.h"

class Clock
{
public:
	Clock();

	void init();
	void getEndTime();
	void getTotalTime();

private:
	clock_t start, finish, first;
};

Clock::Clock()
{
	first = start = clock();
}

inline void Clock::init()
{
	finish = clock();
	cout << "Initialization: " << (double)(finish - start) / CLOCKS_PER_SEC << " sec" << endl;
	start = finish;
}

inline void Clock::getEndTime()
{
	finish = clock();
	cout << "->End (" << (double)(finish - start) / CLOCKS_PER_SEC << " sec)" << endl;
	start = finish;
}

inline void Clock::getTotalTime()
{
	finish = clock();
	cout << "Total: " << ((double)(finish - first)) / ((double)CLOCKS_PER_SEC) << " sec" << endl;
}
