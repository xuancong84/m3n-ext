#ifndef FUN_H
#define FUN_H
#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <algorithm>
#include <math.h>
using namespace std;

const double LOGINF=37;
const double LOGZERO=-LOGINF;
const double INF=1e+37;
bool split_string(char *str, const char *cut, vector<char *> &strs, int mode=0);
char* catch_string(char *str, char *head, char* tail,char* catched);//catch the first substring catched between head and tail in str
char* catch_string(char *str, char* tail, char* catched);//catch the first substring before tail
char* itoa(int value, char* result, int base=10);
ostream *OpenWrite(const string &filename);
istream *OpenRead(const string &filename);
void CloseIO(void *ps);
void CopyFile(const string &fn_in, const string &fn_out);

void forward_backward_viterbi(int ysize,
							  int order,
							  int node_num,
							  vector<double> &lattice,
							  vector<double> &optimum_lattice,
							  vector<vector<int> > &optimum_paths,
							  double head_off=0,
							  double initial_value=0,
							  double min=-INF,
							  double max=INF);

void forward_backward(int ysize,
					  int order,
					  int node_num,
					  vector<double> &lattice,
					  vector<double> &alpha,
					  vector<double> &beta,
					  double &z);

void calculate_margin(int ysize,
							  int order,
							  int node_num,
							  vector<double> &lattice,
							  vector<double> &alpha,
							  vector<double> &beta,
							  double &z,
							  vector<double> &margin);

void viterbi(int node_num, int order, int ysize, int nbest, vector<double>& path, vector<std::vector<int> > &best_path);

template <class T> inline T _min(T x, T y) { return(x < y) ? x : y; }
template <class T> inline T _max(T x, T y) { return(x > y) ? x : y; }

inline double log_sum_exp(double x,double y)
{
    double vmin = _min(x, y);
    double vmax = _max(x, y);
    if (vmax > vmin + LOGINF) {
      return vmax;
    } else {
      return vmax + log(exp(vmin - vmax) + 1.0);
    }
}


template <class T, class cmp_func>
bool vector_search(vector<T> &v, const T & s, int &index, cmp_func cmp)
{
	pair<typename vector< T >::const_iterator, typename vector< T >::const_iterator> IterPair;
	IterPair = equal_range( v.begin( ), v.end( ), s , cmp);
	index=IterPair.first-v.begin();
	if ( IterPair.first == IterPair.second )//not found
		return false;
	return true;
}



template <class T>
bool vector_search(vector<T> &v, const T & s, int &index)
{
	pair<typename vector< T >::const_iterator,typename vector< T >::const_iterator> IterPair;
	IterPair = equal_range( v.begin( ), v.end( ), s );
	index=IterPair.first-v.begin();
	if ( IterPair.first == IterPair.second )//not found
	{
		return false;
	}
	return true;
}

template <class T>
bool vector_insert(vector<T> &v, const T & s, int index)
{
	if(index>v.size())return false;
	typename vector<T>::iterator t=v.begin()+index;
	v.insert(t,s);
	return true;
}


template <class T>
void combine(vector<T> &v, int left, int m, int right, vector<int> &index)
{
	vector<T> tempv(v.begin()+left,v.begin()+right+1);
	vector<int> tempindex(index.begin()+left,index.begin()+right+1);

	int left_size=m-left+1;
	int size=right-left+1;
	int middle=m-left+1;
	int i=0;
	int j=middle;
	int k=left;
	while(i<left_size && j<size)
	{
		if(tempv[i]>=tempv[j])
		{
			v[k]=tempv[i];
			index[k]=tempindex[i];
			k++;
			i++;
		}else{
			v[k]=tempv[j];
			index[k]=tempindex[j];
			k++;
			j++;
		}
	}
	while(i<left_size)
	{
		v[k]=tempv[i];
		index[k]=tempindex[i];
		k++;
		i++;
	}
}


template <class T>
void merge_sort(vector<T> &v, int left, int right, vector<int> &index)
{
    if (left<right)
    {
        int m=(left+right)/2;
        merge_sort(v,left, m,index);
        merge_sort(v,m+1, right, index);
        combine(v,left, m, right,index);
    }
}


template <class T>
void merge_sort(vector<T> v, vector<int> &index)
//index[i] is the original index of i th best element
{
	index.clear();
	index.resize(v.size());
	for(int i=0;i<v.size();i++)
		index[i]=i;
	merge_sort(v,0,v.size()-1,index);
}


template <class T>
class inverse_cmp
{
	public:
	bool operator()(T s, T t) const 
	{
		return s>t;
	}
};

void computeCPR(const vector <int> &ref, const vector <int> &sys, int &C, int &P, int &R);
void computeCPR(int ref, int sys, int &C, int &P, int &R);
void confusion_table(map <int, map<int, int> > &out_table, vector <int> &ref, vector <int> &sys);
int normalize_confusion_table(map <int, map<int, int> > &table, const vector <vector<double> > &ref, vector <vector<double> > &sys, bool skip_zero=false);
void normalize_matrix(const vector <vector<double> > &ref, vector <vector<double> > &sys, bool skip_zero);
double std_deviation(double *buf, int size);
int sum(int *p, int n);
int sum(vector <int> &p);
double KL_div(vector <double> &cannot_be_zero, vector <int> &can_be_zero);
vector <double> & norm_P(vector <double> &out, vector <int> &in, double add_smooth=0);

#endif
