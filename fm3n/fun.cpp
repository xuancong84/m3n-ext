#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <cassert>
#include "fun.h"
#include "gzstream.h"

bool split_string(char *str, const char *cut, vector<char *> &strs, int mode)
{
	size_t size = 0;
	char *p=str,*q;
	bool ret=false;
	if(mode==0){	// label is at the last column
		while (q=strstr(p,cut))
		{
			*q=0;
			strs.push_back(p);
			p=q+strlen(cut);
			ret=true;
		}
		if(p) strs.push_back(p);
	}else{		// label is at 1st column
		q=strstr(p,cut);
		if(!q)
			return false;
		*q=0;
		p=q+strlen(cut);
		while (q=strstr(p,cut))
		{
			*q=0;
			strs.push_back(p);
			p=q+strlen(cut);
			ret=true;
		}
		if(p) strs.push_back(p);
		strs.push_back(str);
	}
	return ret;
}

ostream *OpenWrite(const string &filename){
	if(filename.size()>3) if(filename.substr(filename.size()-3,3)==".gz")
		return new ogzstream(filename.c_str());
	return new ofstream(filename.c_str());
}

istream *OpenRead(const string &filename){
	if(filename.size()>3) if(filename.substr(filename.size()-3,3)==".gz")
		return new igzstream(filename.c_str());
	return new ifstream(filename.c_str());
}

void CloseIO(void *ps){
	ogzstream *po = dynamic_cast<ogzstream*>((ostream*)ps);
	if(po) { delete po; return;}
	igzstream *pi = dynamic_cast<igzstream*>((istream*)ps);
	if(pi) { delete pi; return;}
	ofstream *pO = dynamic_cast<ofstream*>((ostream*)ps);
	if(pO) { delete pO; return;}
	ifstream *pI = dynamic_cast<ifstream*>((istream*)ps);
	if(pI) { delete pI; return;}
}

void CopyFile(const string &fn_in, const string &fn_out){
	ifstream fin(fn_in.c_str(), ios::binary);
	ofstream fout(fn_out.c_str(), ios::binary);

	fout << fin.rdbuf();

	fin.close();
	fout.close();
}

char* catch_string(char *str, char *head, char* tail, char* catched)
{
	char *p=str;
	char *q;
	q=strstr(str,head);
	if(!q)		return NULL;
	p=q+strlen(head);
	q=strstr(p,tail);
	if(!q)		return NULL;
	strncpy(catched,p,q-p);
	catched[q-p]=0;
	return q+strlen(tail);
}

char* catch_string(char *str, char* tail, char* catched)
// catch_string("12345","3",catched) => catched="12", return "45"
{
	char *q;
	q=strstr(str,tail);
	if(!q)
	{
		strcpy(catched,str);
		return NULL;
	}
	strncpy(catched,str,q-str);
	catched[q-str]=0;
	return q+strlen(tail);
}

char* itoa(int value, char* result, int base) {
	// check that the base if valid
	if (base < 2 || base > 36) { *result = '\0'; return result; }

	char* ptr = result, *ptr1 = result, tmp_char;
	int tmp_value;

	do {
		tmp_value = value;
		value /= base;
		*ptr++ = "zyxwvutsrqponmlkjihgfedcba9876543210123456789abcdefghijklmnopqrstuvwxyz" [35 + (tmp_value - value * base)];
	} while ( value );

	// Apply negative sign
	if (tmp_value < 0) *ptr++ = '-';
	*ptr-- = '\0';
	while(ptr1 < ptr) {
		tmp_char = *ptr;
		*ptr--= *ptr1;
		*ptr1++ = tmp_char;
	}
	return result;
}

void forward_backward_viterbi(int ysize,
							  int order,
							  int node_num,
							  vector<double> &lattice,
							  vector<double> &optimum_lattice,		//spaces should be provided
							  //vector<vector<int> > &optimum_paths,	//spaces should be provided
							  vector<int> &alpha_path,	//spaces should be provided
							  vector<int> &beta_path,	//spaces should be provided
							  double head_off,
							  double initial_value,
							  double min,
							  double max){
//calculate max joint probability through each clique
	int i,j,k,ii;
	int node_anum=1;
	for(i=0;i<order;i++) node_anum*=ysize;
	int path_num=node_anum*ysize;
	fill(optimum_lattice.begin(),optimum_lattice.end(),0);
/*
	for(i=0;i<optimum_paths.size();i++)
		fill(optimum_paths[i].begin(),optimum_paths[i].end(),-1);//store optimum path index, initialzed with -1
*/
	alpha_path = vector<int>(node_anum*node_num,-1);		//forward path
	beta_path = vector<int>(node_anum*node_num,-1);		//backward path
	vector<double> alpha(node_anum*node_num,initial_value);
	vector<double> beta(node_anum*node_num,initial_value);
	vector<int> first_cal(node_anum*node_num,1);

	//forward
	for(i=0;i<node_num;i++){
		double *cur_latt=&lattice[path_num*i];
		double *cur_alpha=&alpha[i*node_anum];
		int *cur_first_cal=&first_cal[i*node_anum];
		//cal alpha of current node
		if(i>0){
			const double *last_alpha=&alpha[(i-1)*node_anum];
			for(j=0;j<path_num;j++){
				ii=j % node_anum;//current alpha index
				k=j / ysize;//last alpha index
				double logp=cur_latt[j]+last_alpha[k];
				logp=logp>min?logp:min;
				logp=logp<max?logp:max;
				if(cur_first_cal[ii]||logp>cur_alpha[ii]){
					cur_alpha[ii]=logp;
					alpha_path[node_anum*i+ii]=j;//the best path to cur_alpha[ii] is j
					cur_first_cal[ii]=0;
				}
			}
		}else{
			for(j=0;j<path_num;j++){
				ii=j % node_anum;//current alpha index
				k=j / ysize;//last alpha index
				double logp=cur_latt[j];
				logp+=head_off;
				logp=logp>min?logp:min;
				logp=logp<max?logp:max;
				if(cur_first_cal[ii]||logp>cur_alpha[ii]){
					cur_alpha[ii]=logp;
					alpha_path[node_anum*i+ii]=j;//the best path to cur_alpha[ii] is j
					cur_first_cal[ii]=0;
				}
			}
		}
	}

	//backward
	fill(first_cal.begin(),first_cal.end(),1);
	for(i=node_num-1;i>=0;i--){
		int *cur_first_cal=&first_cal[i*node_anum];
		double *cur_beta=&beta[i*node_anum];
		//calculate beta of last node
		if(i<node_num-1){
			double *last_beta=&beta[(i+1)*node_anum];
			double *last_latt=&lattice[path_num*(i+1)];
			for(j=0;j<path_num;j++){
				k=j % node_anum;//last beta index
				ii=j / ysize;//cur beta index
				double logp=last_beta[k]+last_latt[j];
				logp=logp>min?logp:min;
				logp=logp<max?logp:max;
				if(cur_first_cal[ii]||logp>cur_beta[ii]){
					cur_beta[ii]=logp;
					beta_path[node_anum*i+ii]=j;//the best path to cur_beta[ii] is j
					cur_first_cal[ii]=0;
				}
			}
		}else{
			for(j=0;j<path_num;j++){
				k=j % node_anum;//last beta index
				ii=j / ysize;//cur beta index
				double logp=0;//
				logp=logp>min?logp:min;
				logp=logp<max?logp:max;
				if(cur_first_cal[ii]||logp>cur_beta[ii]){
					cur_beta[ii]=logp;
					beta_path[node_anum*i+ii]=j;//the best path to cur_beta[ii] is j
					cur_first_cal[ii]=0;
				}
			}
		}
	}
	//trace back
	//for each clique

	for(i=0;i<node_num;i++){
		for(j=0;j<path_num;j++){

//			vector<int> &cur_path=optimum_paths[i*path_num+j];
			
			int q=j%node_anum;//current index
			int p=j/ysize;//last index
			//set optimum_lattice
			if(i>0){
				optimum_lattice[i*path_num+j]=alpha[(i-1)*node_anum+p]+lattice[i*path_num+j]+beta[i*node_anum+q];
			}else{
				optimum_lattice[i*path_num+j]=head_off+lattice[i*path_num+j]+beta[i*node_anum+q];
			}
			optimum_lattice[i*path_num+j]=optimum_lattice[i*path_num+j]>min?optimum_lattice[i*path_num+j]:min;
			optimum_lattice[i*path_num+j]=optimum_lattice[i*path_num+j]<max?optimum_lattice[i*path_num+j]:max;
/*
			//add alpha
			for(k=i-1;k>=0;k--){//track alpha
				cur_path[k]=alpha_path[k*node_anum+p];
				p=alpha_path[k*node_anum+p]/ysize;
			}
			//add self
			cur_path[i]=j;
			//add beta
			for(k=i+1;k<node_num;k++){//track beta
				cur_path[k]=beta_path[(k-1)*node_anum+q];
				q=beta_path[(k-1)*node_anum+q]%node_anum;
			}
*/
		}
	}
}

void get_optimum_path(vector <int> &cur_path, int i, int j, int ysize, int node_num, int node_anum, vector <int> &alpha_path, vector <int> &beta_path)
{
	//add alpha
	int q=j%node_anum;	//current index
	int p=j/ysize;		//last index
	cur_path.resize(node_num);
	for(int k=i-1;k>=0;k--){//track alpha
		cur_path[k]=alpha_path[k*node_anum+p];
		p=alpha_path[k*node_anum+p]/ysize;
	}
	//add self
	cur_path[i]=j;
	//add beta
	for(int k=i+1;k<node_num;k++){//track beta
		cur_path[k]=beta_path[(k-1)*node_anum+q];
		q=beta_path[(k-1)*node_anum+q]%node_anum;
	}
}


void calculate_margin(int ysize,
							  int order,
							  int node_num,
							  vector<double> &lattice,
							  vector<double> &alpha,
							  vector<double> &beta,
							  double &z,
							  vector<double> &margin){
	int i,j;
	int node_anum=1;
	for(i=0;i<order;i++) node_anum*=ysize;
	int path_num=node_anum*ysize;
	double head_off=-log((double)node_anum);
	if(margin.size()<node_num*path_num)
		margin.resize(node_num*path_num);
	for(i=0;i<node_num;i++)
	{
		double *cur_path=&lattice[path_num*i];
		double *cur_beta=&beta[node_anum*i];
		if(i>0)
		{
			double *last_alpha=&alpha[node_anum*(i-1)];
			for(j=0;j<path_num;j++){
				margin[i*path_num+j]=exp(cur_path[j] + last_alpha[j/ ysize] + cur_beta[j% node_anum] - z);
			}
		}else{//first node
			for(j=0;j<path_num;j++){
				margin[i*path_num+j]=exp(cur_path[j] + head_off + cur_beta[j% node_anum] - z);
			}
		}
	}
}
void forward_backward(int ysize,
							  int order,
							  int node_num,
							  vector<double> &lattice,
							  vector<double> &alpha,
							  vector<double> &beta,
							  double &z){
	int i,j,k,ii;
	int node_anum=1;
	for(i=0;i<order;i++) node_anum*=ysize;
	int path_num=node_anum*ysize;
	if(alpha.size()<node_anum*node_num){
		alpha.resize(node_anum*node_num);
		beta.resize(node_anum*node_num);
	}
	double head_off=-log((double)node_anum);
	//forward

	fill(alpha.begin(),alpha.end(),0);
	vector<int> first_cal(node_anum*node_num,1);
	for(i=0;i<node_num;i++)
	{
		double *cur_path=&lattice[path_num*i];
		//cal alpha of current node
		if(i>0)
		{
			double *cur_alpha=&alpha[i*node_anum];
			const double *last_alpha=&alpha[(i-1)*node_anum];
			int *cur_first=&first_cal[i*node_anum];
			for(j=0;j<path_num;j++)
			{
				ii=j % node_anum;
				k=j / ysize;
				if(!cur_first[ii])
				{
					cur_alpha[ii]=log_sum_exp(last_alpha[k]+cur_path[j],cur_alpha[ii]);
				}else{
					cur_alpha[ii]=last_alpha[k]+cur_path[j];
					cur_first[ii]=0;
				}
			}
		}else{
			double *cur_alpha=&alpha[i*node_anum];
			int *cur_first=&first_cal[i*node_anum];
			for(j=0;j<path_num;j++)
			{
				ii=j % node_anum;
				if(!cur_first[ii])
				{
					cur_alpha[ii]=log_sum_exp(cur_path[j]+ head_off ,cur_alpha[ii]);
				}else{
					cur_alpha[ii]=cur_path[j]+ head_off ;
					cur_first[ii]=0;
				}
			}
		}
	}

	//backward
	
	fill(beta.begin(),beta.end(),0);
	fill(first_cal.begin(),first_cal.end(),true);
	vector<double> last_path(path_num,0);
	for(i=node_num-1;i>=0;i--)
	{
		//calculate beta of last node
		if(i<node_num-1)
		{
			double *cur_beta=&beta[i*node_anum];
			double *last_beta=&beta[(i+1)*node_anum];
			int *cur_first=&first_cal[i*node_anum];
			double *last_path=&lattice[path_num*(i+1)];
			for(j=0;j<path_num;j++)
			{
				k=j % node_anum;
				ii=j / ysize;
				if(!cur_first[ii])
				{
					cur_beta[ii]=log_sum_exp(last_beta[k]+last_path[j],cur_beta[ii]);
				}else{
					cur_beta[ii]=last_beta[k]+last_path[j];
					cur_first[ii]=0;
				}
			}
		}else{
			double *cur_beta=&beta[i*node_anum];
			for(j=0;j<node_anum;j++)
				cur_beta[j]=0;
		}
	}
	//calculate z(x)
	z=alpha[node_anum*(node_num-1)];
	for(i=1;i<node_anum;i++)
		z=log_sum_exp(z, alpha[node_anum*(node_num-1)+i]);
}


void viterbi(int node_num, int order, int ysize, int nbest, vector<double>& path, vector<std::vector<int> > &best_path)
//input parameter: node_num, order, ysize, nbest, path
//output parameter: best_path
//node_num: length of sequence, i.e, number of nodes
//order: order of viterbi algorithm, e.g. order = 1 means  first order viterbi algoritm
//ysize: label size, label indexes are from 0 to ysize - 1
//nbest: output top nbest paths
//path: double path[ node_num * ysize ^ (order + 1) ], cost of each path segment
//		path[i * ysize ^ (k + 1) + j1 * ysize ^ k + ... + jk], j2, ... ,jk < ysize; i=0,..., node_num-1
//		denotes the segment through the i , ... i + k - 1 th node with labels j1, ... ,jk respectively
//		e.g., in first order viterbi, path[i * ysize ^ 2 + j1 * ysize + j2]
//      denote the cost of segment from i th node with label j1 to i + 1 th node with label j2
//best_path: indexes of labels of the top best paths, best_path[i] is the i th best path, i < nbest
{
	int i,j;
	int i1,j1,k1,i2,j2,k2;
	//last node: i1 th node, j1 th tag, k1 th best
	//current node: i2 th node, j2 th tag, k2 th best
	int node_anum=(int)pow((double)ysize,order);
	int path_num=node_anum*ysize;
	std::vector<std::vector<double> >last_best(node_anum);
	// i th tag, j th best: last_best[i][j]
	std::vector<std::vector<double> >cur_best(node_anum);
	std::vector<std::vector<std::vector< std::pair<int, int> > > >best_prev(node_num+1);
	//best_prev of i th node j th tag, k th best: best_prev[i][j][k]
	std::vector<std::vector<std::vector< int > > >best_link(node_num+1);
	//index of the path from i th node j th tag, k th best to i-1 th node best_prev[i][j][k].first th tag, best_prev[i][j][k].second th best
	std::vector<double> final_path(node_anum,0);
	for(i=0;i<=node_num;i++)
	{
		i2=i;
		//current node: i2 th node
		i1=i-1;
		//last node: i1 th node
		double *cur_path;
		if(i<node_num)
		{
			best_prev[i].resize(node_anum);
			best_link[i].resize(path_num);
			cur_path=&path[path_num*i];
		}else{
			best_prev[i].resize(1);
			best_link[i].resize(1);
			cur_path=&final_path[0];
		}

		//search n-best path
		if(i>0)
		{
			last_best=cur_best;
			for(j=0;j<node_anum;j++)
				cur_best[j].clear();
			if(i==node_num)
				cur_best.resize(1);
		}else{
			std::vector<double> init_best(1,0);
			last_best[0]=init_best;
		}
		int routine_num=i<node_num?path_num:node_anum;
		for(j=0;j<routine_num;j++)
		{
			if(i<node_num)
			{	
				j2 = j % node_anum;//current node, j2 th tag
				j1 = j / ysize;//last node, j1 th tag
			}else{
				j2 = 0;
				j1 = j;
			}
			for(k1=0;k1<last_best[j1].size();k1++)
			{
				double cost=last_best[j1][k1];
				cost+=cur_path[j];
				//last node : j1 th tag,  k1 th best
				std::pair<std::vector< double >::const_iterator,std::vector< double >::const_iterator> ip;
				ip = std::equal_range( cur_best[j2].begin( ), cur_best[j2].end( ), cost, inverse_cmp<double>());
				k2=ip.first-cur_best[j2].begin();
				//current node : j2 th tag, k2 th best, now, search for k2
				if(k2<nbest)
				{
					//vector_insert(cur_best[j2],cost,k2);
					cur_best[j2].insert(cur_best[j2].begin()+k2,cost);
					std::pair<int,int> q=std::make_pair(j1,k1);
					best_prev[i2][j2].insert(best_prev[i2][j2].begin()+k2,q);
					best_link[i2][j2].insert(best_link[i2][j2].begin()+k2,j);
					//for current node, i.e. i2 th node, its j2 th tag, k2 th 
					//best previous is the j1 th tag, k1 th best of last node
					if(cur_best[j2].size()>nbest)//drop the nbest+1 th candidate
					{
						cur_best[j2].pop_back();
						best_prev[i2][j2].pop_back();
						best_link[i2][j2].pop_back();
					}
				}else{
					break;
				}
			}
		}
	}

	for(i=0;i<best_prev[node_num][0].size();i++)
	{
		std::vector<int> bst_path(node_num);
		k2=i;
		j2=0;
		for(i2=node_num;i2>0;i2--)
		{
			std::pair<int ,int> p=best_prev[i2][j2][k2];
			j2=p.first;
			k2=p.second;
			bst_path[i2-1]=best_link[i2-1][j2][k2] % ysize;
		}
		best_path.push_back(bst_path);
	}
}

void computeCPR(int ref, int sys, int &C, int &P, int &R){
	if(ref<=3)
		P++;
	if(ref<=3 && sys<=3)
		C++;
	if(sys<=3)
		R++;
}

void computeCPR(const vector <int> &ref, const vector <int> &sys, int &C, int &P, int &R)
{
	int len = ref.size();
	if(len!=sys.size())
		exit(1);
	for(int x=0; x<len; x++)
		computeCPR(ref[x], sys[x], C, P, R);
}

void confusion_table(map <int, map<int, int> > &out_table, vector <int> &ref, vector <int> &sys)
{
	int len = ref.size();
	if(len!=sys.size())
		exit(1);
	for(int x=0; x<len; x++)
		++out_table[ref[x]][sys[x]];
}

int normalize_confusion_table(map <int, map<int, int> > &table, const vector <vector<double> > &ref, vector <vector<double> > &sys, bool skip_zero)
{
	double ref_sum=0, sys_sum=0;
	for(int x=0; x<ref.size(); x++)
		for(int y=0; y<ref[x].size(); y++){
			if(skip_zero) if(sys[x][y]==0)
				continue;
			ref_sum += table[x][y]*ref[x][y];
			sys_sum += table[x][y]*sys[x][y];
		}
	if(!ref_sum)
		return 1;
	if(!sys_sum)
		return -1;
	double ratio = ref_sum/sys_sum;
	for(int x=0; x<ref.size(); x++)
		for(int y=0; y<ref[x].size(); y++)
			sys[x][y] *= ratio;
	return 0;
}

void normalize_matrix(const vector <vector<double> > &ref, vector <vector<double> > &sys, bool skip_zero)
{
	double ref_sum=0, sys_sum=0;
	int ref_cnt=0, sys_cnt=0;
	for(int x=0; x<ref.size(); x++)
		for(int y=0; y<ref[x].size(); y++){
			if(!skip_zero || ref[x][y]!=0){
				ref_sum += ref[x][y]*ref[x][y];
				ref_cnt ++;
			}
			if(!skip_zero || sys[x][y]!=0){
				sys_sum += sys[x][y]*sys[x][y];
				sys_cnt ++;
			}
		}
	double ratio = sqrt(ref_sum/ref_cnt)/sqrt(sys_sum/sys_cnt);
	for(int x=0; x<ref.size(); x++)
		for(int y=0; y<ref[x].size(); y++)
			sys[x][y] *= ratio;
}

double std_deviation(double *buf, int size)
{
	double ex=0, ex2=0;
	for(int x=0; x<size; x++){
		ex += buf[x];
		ex2 += buf[x]*buf[x];
	}
	ex /= size;
	ex2 /= size;
	return sqrt((ex2-ex*ex)*size/(size-1));
}

int sum(int *p, int n){
	int s=0;
	for(int x=0; x<n; x++)
		s+=p[x];
	return s;
}

int sum(vector <int> &p){
	int s=0;
	for(int x=0; x<p.size(); ++x)
		s+=p[x];
	return s;
}

double KL_div(vector <double> &cannot_be_zero, vector <int> &can_be_zero){
	assert (cannot_be_zero.size()==can_be_zero.size());
	int n=cannot_be_zero.size();
	double sum1=sum(can_be_zero);
	double ret=0;
	for(int x=0; x<n; ++x){
		if(can_be_zero[x]){
			double p = can_be_zero[x]/sum1;
			ret+=log(p/cannot_be_zero[x])*p;
		}
	}
	return ret;
}

vector <double> & norm_P(vector <double> &out, vector <int> &in, double add_smooth){
	int len = in.size();
	double mul = 1.0/(sum(in)+add_smooth*len);
	out.resize(in.size());
	for(int x=0; x<in.size(); ++x)
		out[x] = (in[x]+add_smooth)*mul;
}



