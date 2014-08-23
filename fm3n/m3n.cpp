#include <fstream>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <cstdlib>
#include <assert.h>
#include <time.h>
#include <ctime>
#include <algorithm>
#ifndef _WIN32
#include <unistd.h>
#endif
#include "fun.h"
#include "m3n.h"
/* dual problem:
	min_a 1/2 C||\sum_{X,Y} a_X(Y)Df(X,Y)||_2^2 - \sum_{X,Y} a_X(Y) Dt(X,Y)
	s.t.	\sum_Y a_X(Y) = 1
			a_X(Y)>=0,      \forall X,Y
*/

const int MAXSTRLEN=8192;
const int PAGESIZE=8192;
const double EPS=1e-12;
using namespace std;

int int_cmp(const void *a,const void *b){
	return *(int *)a - *(int *)b; 
}

int str_comp_func(const void* s, const void* t)
{
	return strcmp((const char*)s, (const char*)t);
}

void ee(char *str){
	cerr << str << endl;
	exit(0);
}

char *get_pure_label(char *label, bool &mark){
	if(label[0]==weighted_node_sym){
		mark=1;
		return &label[1];
	}
	mark=0;
	return label;
}

FM3N::FM3N(){
	read_mode=0;
	print_feature_file=NULL;

	_eta=0.000001;
	_C=1;
	_B=2;
	_freq_thresh=0;
	_order=0;
	_nbest=1;
	_margin=0;
	_tag_str.set_size(PAGESIZE);
	_x_str.set_size(PAGESIZE);
	_nodes.set_size(PAGESIZE);
	_cliques.set_size(PAGESIZE);
	_clique_node.set_size(PAGESIZE);
	_node_clique.set_size(PAGESIZE);
	_clique_feature.set_size(PAGESIZE);
	_kernel_type=LINEAR_KERNEL;
	_get_kernel=&FM3N::linear_kernel;
	_get_kernel_list=&FM3N::linear_kernel;
	_kernel_s=1;	// poly_kernel = (_kernel_s*linear_kernel+_kernel_r)^_kernel_d
	_kernel_d=1;	// neural_kernel=tanh(_kernel_s*linear_kernel+_kernel_r)
	_kernel_r=0;	// rbf_kernel = exp(-_kernel_s*norm2)
	_version=11;
	_test_nodes.set_size(PAGESIZE);
	_test_cliques.set_size(PAGESIZE);
	_test_clique_node.set_size(PAGESIZE);
	_test_node_clique.set_size(PAGESIZE);
	_test_clique_feature.set_size(PAGESIZE);

	_mu=NULL;
	_mu_size=0;
#ifndef SPARSE_W
	_w=NULL;
#endif
	_w_size=0;
	_y_max_id=0;
	_max_iter=10;
	_obj=0;
	_print_level=0;
}

FM3N::~FM3N(){
	if(_mu){
		delete [] _mu;
		_mu=NULL;
	}
#ifndef SPARSE_W
	if(_w){
		delete [] _w;
		_w=NULL;
	}
#endif
}

bool FM3N::set_para(char *para_name, char *para_value){
	if(!strcmp(para_name,"C"))
		_C=atof(para_value);
	else if(!strcmp(para_name,"B"))
		_B=atof(para_value);
	else if(!strcmp(para_name,"freq_thresh"))
		_freq_thresh=atoi(para_value);
	else if(!strcmp(para_name,"nbest"))
		_nbest=abs(atoi(para_value));
	else if(!strcmp(para_name,"eta"))
		_eta=atof(para_value);
	else if(!strcmp(para_name,"kernel_type")){
		_kernel_type=atoi(para_value);
		if(_kernel_type==LINEAR_KERNEL){
			_get_kernel=&FM3N::linear_kernel;
			_get_kernel_list=&FM3N::linear_kernel;
		}else if(_kernel_type==POLY_KERNEL){
			_get_kernel=&FM3N::poly_kernel;
			_get_kernel_list=&FM3N::poly_kernel;
		}else if(_kernel_type==NEURAL_KERNEL){
			_get_kernel=&FM3N::neural_kernel;
			_get_kernel_list=&FM3N::neural_kernel;
		}else if(_kernel_type==RBF_KERNEL){
			_get_kernel=&FM3N::rbf_kernel;
			_get_kernel_list=&FM3N::rbf_kernel;
		}else{
			cerr<<"incorrect kernel_type"<<endl;
			return false;
		}
	}else if(!strcmp(para_name,"kernel_s")){
		_kernel_s=atof(para_value);
	}else if(!strcmp(para_name,"kernel_d")){
		_kernel_d=atoi(para_value);
	}else if(!strcmp(para_name,"kernel_r")){
		_kernel_r=atof(para_value);
	}else if(!strcmp(para_name,"max_iter")){
		_max_iter=atoi(para_value);
	}else if(!strcmp(para_name,"print_level")){
		_print_level=atoi(para_value);
	}else if(!strcmp(para_name,"margin")){
		_margin=atoi(para_value);
	}else
		return false;
	return true;
}
bool FM3N::load_loss(string &firstline, istream &ifs){
	return true;
}
bool FM3N::load_loss(const string& loss_file){
	if(loss_file.empty()){
		_loss.clear();
		for(int z=0; z<=_y_max_id; ++z){
			int ysize=_ysizes[z];
			vector <vector <double> > loss(ysize, vector<double>(ysize,1.0));
			for(int x=0; x<ysize; x++)
				loss[x][x]=0;
			_loss.push_back(loss);
		}
		return true;
	}
	istream &ifs=*OpenRead(loss_file);
	if(!ifs.good()){
		cerr<<"can not open file "<<loss_file<<endl;
		return false;
	}

	_loss.resize(_y_max_id+1);
	for(int z=0; z<=_y_max_id; ++z){

		char line[1024*8];
		ifs.getline(line,sizeof(line)-1);
		if(line[0]=='%'){
			_loss_type=&line[1];
			ifs.getline(line,sizeof(line)-1);
		}else
			_loss_type="";

		map <string,int> labs2int;
		vector <vector <double> > loss;
		while(!ifs.eof()){
			ifs.getline(line,sizeof(line)-1);
			vector<char *> v;
			if(!split_string(line,"\t",v))
				continue;
			int a = labs2int.size();
			labs2int[v[0]] = a;
			loss.push_back(vector<double>());
			for(int x=1; x<v.size(); ++x)
				loss.back().push_back(atof(v[x]));
		}
		CloseIO(&ifs);

		// map labels
		int ysize = _ysizes[z];
		_loss[z].resize(ysize, vector<double>(ysize));
		for(int x=0; x<ysize; ++x)
			for(int y=0; y<ysize; ++y)
				_loss[z][x][y] = loss[labs2int[_tags[z][x]]][labs2int[_tags[z][y]]];

		cerr << "loss-type for layer " << z << " : " << _loss_type << endl;
		cerr << "loss-matrix for layer " << z << " : " << endl;
		for(int x=0; x<_loss[z].size(); x++){
			cerr << _tags[z][x];
			for(int y=0; y<_loss[z][x].size(); y++)
				cerr << '\t' << _loss[z][x][y];
			cerr << endl;
		}
	}
	return true;
}
/*
void view_lattice(const vector <double> &lat, int len, int N){
	vector <vector <vector<double> > > view;
	vector <vector<double> > m55;
	for(int x=0; x<N; x++)
		m55.push_back(vector <double> (5));
	for(int x=0; x<len; x++)
		view.push_back(m55);
	for(int x=0; x<len; x++)
		for(int y=0; y<N; y++)
			for(int z=0; z<N; z++)
				view[x][y][z] = lat[x*N*N+y*N+z];
	int a=5;
	a+=a;
}*/
bool FM3N::learn(const string& templet_file, const string& training_file,
		const string& model_file, const string& loss_file, bool relearn){
	cerr<<"pocket M3N"<<endl<<"version 0."<<_version<<endl<<"Copyright(c)2008 Media Computing and Web Intelligence LAB, Fudan Univ.\nAll rights reserved"<<endl;
	int i;
	_model=LEARN_MODEL;
	if(relearn){
		if(!load_model(model_file))
			return false;
		_templets.clear();
		_feature2trellis.clear();
		_trellis2feature.clear();
		_gsize=0;
		_order=0;
		_cols=0;
		_ysize=0;
		_path_num=0;
		_node_anum=0;
		_head_offset=0;
		_tags.clear();
		_tag_str.clear();
		_xindex.clear();
		_x_str.clear();
		_sequences.clear();
		_nodes.clear();
		_cliques.clear();
		_clique_node.clear();
		_node_clique.clear();
		_clique_feature.clear();
	}
	if(!load_templet(templet_file))
		return false;
	cerr<<"templates loaded"<<endl;
	if(!check_training(training_file))
		return false;
	if(!load_loss(loss_file))
		return false;
	_w_size=0;//lambda size
	if(!generate_features(training_file))
		return false;
	if(print_feature_file){
		print_feature_file->flush();
		delete print_feature_file;
		cerr << "Features printed to file successfully!" << endl;
		exit(0);
	}
	shrink_feature();
#ifdef SPARSE_W
	{
		// build _xend map
		_xend.clear();
		for(map<char *, FeatureInfo, str_cmp>::iterator it= _xindex.begin(); it!=_xindex.end(); ++it){
			FeatureInfo &fi = it->second;
			_xend[fi.global_id] = fi.global_id+_sizes[fi.clique_type];
		}
	}
#endif
	for(i=0;i<_sequences.size();i++)
		sort_feature(_sequences[i]);
	vector<sequence>(_sequences).swap(_sequences);

	//write model part 1
	initialize();
	write_model(model_file,true);

	cerr<<"sequence number: "<<_sequences.size()<<endl  <<"feature number: "<<_xindex.size()<<endl  <<"parameter number: "<<_w_size<<endl
			<<"C: "<<_C<<endl<<"B: "<<_B<<endl<<"freq_thresh: "<<_freq_thresh<<endl<<"eta: "<<_eta<<endl<<"max_iter: "<<_max_iter<<endl;

	_tags.clear();
	_tag_str.clear();
	_xindex.clear();
	_x_str.clear();
	_templets.clear();
	
	if(_kernel_type==LINEAR_KERNEL)
		cerr<<"parameter number: "<<_w_size<<endl;
	else
		cerr<<"parameter number: "<<_mu_size<<endl;
	switch(_kernel_type){
		case LINEAR_KERNEL: cerr<<"linear kernel: k(a,b)=<a,b>"<<endl;break;
		case POLY_KERNEL: cerr<<"polynomial kernel: k(a,b)=("<<_kernel_s<<"*<a,b>+"<<_kernel_r<<")^"<<_kernel_d<<endl;break;
		case RBF_KERNEL: cerr<<"rbf kernel: k(a,b)=exp{-"<<_kernel_s<<"*||a-b||^2}"<<endl;break;
		case NEURAL_KERNEL: cerr<<"neural kernel: k(a,b)=tanh("<<_kernel_s<<"*<a,b>+"<<_kernel_r<<")"<<endl;break;
	}
	
	int max_seq_len=0;
	for(i=0;i<_sequences.size();i++){
		if(max_seq_len<_sequences[i].node_num)
			max_seq_len=_sequences[i].node_num;
	}
	_alpha_lattice.resize(max_seq_len*_path_num);
	_v_lattice.resize(max_seq_len*_path_num);
	_optimum_alpha_lattice.resize(max_seq_len*_path_num);
/*
	_optimum_alpha_paths.resize(max_seq_len*_path_num);
	for(i=0;i<_optimum_alpha_paths.size();i++)
		_optimum_alpha_paths[i].resize(max_seq_len);
*/
	_optimum_v_lattice.resize(max_seq_len*_path_num);
/*
	_optimum_v_paths.resize(max_seq_len*_path_num);
	for(i=0;i<_optimum_v_paths.size();i++)
		_optimum_v_paths[i].resize(max_seq_len);
*/
	int iter=0;
	double kkt_violation;
	double diff=1;
	int converge=0;
	clock_t start_time=clock();

	int C, P, R;
	double N;

	for(iter=0;iter<_max_iter;){

		//pass through sequences
		double old_obj=_obj;
		clock_t tm = clock();
		for(i=0;i<_sequences.size();i++){
			clock_t tm1 = clock();
			if(tm1-tm>CLOCKS_PER_SEC){
				cerr << "iter: "<< iter << " : " << i << "/" << _sequences.size() <<"\r";
				tm = tm1;
			}
			build_alpha_lattice(_sequences[i]);
			//view_lattice(_alpha_lattice, _sequences[i].node_num, _node_anum);
			build_v_lattice(_sequences[i]);
			//view_lattice(_v_lattice, _sequences[i].node_num, _node_anum);
			if(find_violation(_sequences[i],kkt_violation))
				smo_optimize(_sequences[i]);
			if(_print_level>0)
				fprintf(stderr,"\tseq: %d kkt_violation: %lf\n",i,kkt_violation);
		}
		
/*
		if(iter==9){
			iter+=2;
			--iter;
			--iter;
		}
*/
		if(iter)
			diff=fabs((old_obj-_obj)/old_obj);
		fprintf(stderr,"iter: %d diff: %lf obj: %lf\n",iter,diff,_obj);
		iter++;
		if(diff<_eta)
			converge++;
		else
			converge=0;
		if(converge==3)
			break;

		if(!iter_model_file.empty()){
			char str[100];
			CopyFile(model_file+".tmp.gz", iter_model_file+"."+itoa(iter,str)+".gz.tmp.gz");
			write_model(iter_model_file+"."+itoa(iter,str)+".gz", false);
		}
	}
	double elapse = static_cast<double>(clock() - start_time) / CLOCKS_PER_SEC;
	cerr<<"elapse: "<<elapse<<" s"<<endl;
	//write model part 2
	write_model(model_file,false);
	return true;
}
bool FM3N::write_model(const string &model_file, bool first_part){
	if(first_part){
		string tmp_model_file=model_file+".tmp.gz";
		ostream &fout=*OpenWrite(tmp_model_file);
		if(!fout.good()){
			cerr << "can not open model file: " << model_file << endl;
			return false;
		}
		int i,j,k,ii;
		//write version
		fout<<"version\t"<<_version<<endl;
		//write kernel_type
		fout<<_kernel_type<<endl;
		//write kernel parameters
		fout<<_kernel_s<<endl;
		fout<<_kernel_d<<endl;
		fout<<_kernel_r<<endl;

		//write eta
		fout<<_eta<<endl;
		//write C
		fout<<_C<<endl;
		//write freq_thresh
		fout<<_freq_thresh<<endl;

		//write templets
		for(i=0;i<_templets.size();i++)
			fout<<_templets[i].full_str<<endl;
		fout<<endl;

		//write y
		fout<<_y_max_id<<endl;
		for(int x=0; x<=_y_max_id; ++x){
			fout<<_ysizes[x]<<endl;
			for(i=0;i<_tags[x].size();i++)
				fout<<_tags[x][i]<<endl;
		}
		fout<<endl;

		//write x
		fout<<_cols<<endl<<endl;
		fout<<_xindex.size()<<endl;
		map<char*, FeatureInfo, str_cmp>::iterator it;
		for(it = _xindex.begin(); it != _xindex.end(); it++){
			//fout<<it->first<<'\t'<<it->second<<endl;
			FeatureInfo &info = it->second;
			fout<<it->first<<'\t'<<info.global_id<<'\t'<<info.clique_type<<'\t'<<info.freq<<'\t'<<info.templet_id<<endl;
		}
		fout<<endl;
		if(_kernel_type!=LINEAR_KERNEL){
			//write _sequences info
			fout<<_sequences.size()<<endl;//sequence number
			for(i=0;i<_sequences.size();i++){
				sequence &seq=_sequences[i];
				fout<<seq.node_num<<'\t'<<seq.mu_index<<endl;//sequence node_num, mu_index
				for(j=0;j<seq.node_num;j++){
					node &nod=seq.nodes[j];
					fout<<nod.key<<'\t'<<nod.clique_num<<endl;//node key, clique num
					//for each clique, 	output groupid, feature_num, node_num, key, fvector
					for(k=0;k<nod.clique_num;k++){
						if(!nod.cliques[k])
							continue;
						clique &cli=*(nod.cliques[k]);
						fout<<cli.groupid<<'\t'<<cli.feature_num<<'\t'<<cli.node_num<<'\t'<<cli.key<<endl;
						for(ii=0;ii<cli.feature_num-1;ii++)
							fout<<cli.fvector[ii]<<'\t';
						fout<<cli.fvector[ii]<<endl;
					}
				}
			}
			fout<<endl;
		}
		CloseIO(&fout);
	}else{
		//copy tmp_model_file
		string tmp_model_file = model_file + ".tmp.gz";
		char line[MAXSTRLEN];
		istream &fin = *OpenRead(tmp_model_file);
		ostream &fout= *OpenWrite(model_file);

		while(true){
			fin.getline(line,MAXSTRLEN-1);
			if(!fin.eof())
				fout<<line<<endl;
			else
				break;
		}
		unlink(tmp_model_file.c_str());

		//write obj
		int i;

		fout<<_obj<<endl;
		if(_kernel_type==LINEAR_KERNEL){
			//write w
#ifdef SPARSE_W
			fout<<_w.size()<<endl;
			for(map<int,double>::iterator it=_w.begin(); it!=_w.end(); ++it){
				fout << it->first << " " << it->second << endl;
			}
#else
			fout<<_w_size<<endl;
			for(i=0;i<_w_size;i++){
				fout<<_w[i]<<endl;
			}
#endif
			fout<<endl;
		}
		//write mu
		fout<<_mu_size<<endl;
		for(i=0;i<_mu_size;i++){
			fout<<_mu[i]<<endl;
		}
		CloseIO(&fout);
		CloseIO(&fin);
	}
	return true;
}


bool FM3N::add_templet(char *line){
	if(!line[0]||line[0]=='#')	// skip comments
		return false;

	templet n;
	n.func=0;
	n.mask=0;
	n.full_str=line;
	n.freq_thresh=_freq_thresh;

	// process header
	vector <char*> parts;
	split_string(line, " ", parts);
	for(int x=1; x<parts.size(); x++){
		vector <char*> var_val;
		split_string(parts[x],"=",var_val);
		if(var_val.size()==2){
			if(string(var_val[0])=="freq_thresh")
				n.freq_thresh=atoi(var_val[1]);
		}else if(var_val.size()==1){
			if(string(var_val[0])=="no_header")
				n.mask |= 1;
			if(string(var_val[0])=="sort_x")
				n.mask |= 2;
		}
	}

	char *p=line,*q;
	char word[1000];
	char index_str[1000];
	int index1,index2;

	// process x
	while(q=catch_string(p,"%x[",word)){
		p=q;

		// extract command
		if(n.words.empty()){
			if(!strncmp(word,"%==",3)){			// test equal without identity
				n.func=1;
				strcpy(index_str, &word[3]);
				strcpy(word, index_str);
			}else if(!strncmp(word,"%~=",3)){		// equal with identity
				n.func=2;
				strcpy(index_str, &word[3]);
				strcpy(word, index_str);
			}
		}
		n.words.push_back(word);

		p=catch_string(p,"]",index_str);
		vector <char*> segs;
		split_string(index_str, ",", segs);
		index1=atoi(segs[0]);
		index2=atoi(segs[1]);
		n.x.push_back(make_pair(index1,index2));
		if(segs.size()==3){		// %x[-2,0,5!=FL], col 0 of 2nd previous row with col5!=FL
			n.x.back().first = (1+abs(index1))*(segs[0][0]=='-'?-1:1);
			n.skip_words.resize(n.x.size());
			n.skip_cols.resize(n.x.size());
			string cmd = segs[2];
			char cmd_str[1000];
			strcpy(cmd_str,segs[2]);
			segs.clear();
			int mode;
			if(cmd.find("==")!=string::npos){
				mode=0;
				split_string(cmd_str,"==",segs);
			}else if(cmd.find("!=")!=string::npos){
				mode=0x80000000;
				split_string(cmd_str,"!=",segs);
			}else{
				cerr<<"templet: "<<line<<" incorrect, unknown x command "<<cmd<<endl;
				return false;
			}
			n.skip_words.back() = segs[1];
			n.skip_cols.back() = (mode|atoi(segs[0]));
		}
	}
	q=catch_string(p,"%y[",word);
	if(!q){
		cerr<<"templet: "<<line<<" incorrect"<<endl;
		return false;
	}
	n.words.push_back(word);

	// process y
	p=q-3;
	n.y_max_history=0;
	while(p=catch_string(p,"%y[","]",index_str)){
		vector <char*> ys;
		split_string(index_str,",",ys);
		index1=atoi(ys[0]);
		index2=(ys.size()>1?atoi(ys[1]):0);
		assert(index1<=0 && index2>=0);
		if(index1==-2)
			index1=-2;
		n.y.push_back(make_pair(index1,index2));
		if(index1<n.y_max_history)
			n.y_max_history=index1;
		if(-index1>_order)
			_order=-index1;
		if(index2>_y_max_id)
			_y_max_id=index2;
	}

	// determine clique type id
	sort(n.y.begin(), n.y.end());
	auto it = _ys2cliqueType.find(n.y);
	if(it==_ys2cliqueType.end()){
		n.clique_type = _ys2cliqueType.size();
		_ys2cliqueType[n.y] = n.clique_type;
		_clique_types.push_back(n.y);
		_clique_hist_size.push_back(n.y_max_history);
		assert(_ys2cliqueType.size()==_clique_types.size());
	}else
		n.clique_type = it->second;

	// add templet to list
	_templets.push_back(n);

	return true;
}

bool FM3N::load_templet(const string& templet_file){
	istream &fin=*OpenRead(templet_file);
	//read template
	if (!fin.good()){
		cerr<<"template file: "<<templet_file<<" not found"<<endl;
		return false;
	}
	int cur_group;
	while(!fin.eof()){
		char line[MAXSTRLEN];
		fin.getline(line,MAXSTRLEN-1);
		add_templet(line);
	}
	CloseIO(&fin);
	_gsize=_order+1;
	_feature2trellis.resize(_gsize);
	
	if(_order<0)// no _templets
		return false;
	return true;
}

bool FM3N::check_training(const string& training_file){
	istream &fin=*OpenRead(training_file);
	if(!fin.good())	return false;
	char line[MAXSTRLEN];
	int lines=0;
	_cols=0;
	_tags=vector <vector<char *> >(_y_max_id+1, vector<char*>());
	fprintf(stderr,"Check training ...(per 10000 lines)\n");
	while(!fin.eof()){//fgets(line,MAXSTRLEN-1,fp))
		fin.getline(line,MAXSTRLEN-1);
		lines++;
		if(!line[0]) continue;
		vector <char*> columns;
		if(!split_string(line,"\t",columns,read_mode)){
			cerr<<"columns should be greater than 1"<<endl;
			CloseIO(&fin);
			return false;//columns should be greater than 1
		}
		if(_cols && _cols!=columns.size()){//incompatible
			cerr<<"line: "<<lines<<" columns incompatible"<<endl;
			CloseIO(&fin);
			return false;
		}
		_cols=columns.size();

		for(int y_id=0; y_id<=_y_max_id; ++y_id){
			char *t=columns[_cols-1-y_id];//tag
			if(*t==weighted_node_sym)
				++t;
			int index;
			if(!vector_search(_tags[y_id],t,index,str_cmp())){
				char *p=_tag_str.push_back(t);//copy string
				vector_insert(_tags[y_id],p,index);
			}
		}

		if(!(lines%10000)){
			fprintf(stderr,".");
			fflush(stderr);
		}
	}
	CloseIO(&fin);
	// set _ysizes and _ysize
	_ysizes=vector<int>(_tags.size());
	for(int x=0; x<_tags.size(); ++x){
		_ysizes[x] = _tags[x].size();
		fprintf(stderr,"Number of labels at layer %d = %d\n",x,_ysizes[x]);
	}
	set_group();
	fprintf(stderr,"Total number of labels = %d\n",_ysize);
	return true;
}
inline int FM3N::trellisID2featureID(vector <int> &fids, int tid, vector <pair<int,int> > &y){
	vector <int> Y_hist(_gsize);				// get history of composed labels
	key2hist(&Y_hist[0], tid);
	vector <vector<int> > y_hist(_gsize);		// get history of decomposed labels
	for(int z = 0; z < _gsize; ++z)
		get_decomposed_label(y_hist[z], Y_hist[z]);
	int fid = 0;								// feature joint id
	fids.resize(y.size());						// feature split id
	for(int z = 0; z < y.size(); ++z){
		fids[z] = y_hist[y[z].first+_order][y[z].second];
		fid = fid*_ysizes[y[z].second] + fids[z];
	}
	return fid;
}
void FM3N::set_group(){
	//calculate _templet_group, set the size of each group
	//_order and _tags[*].size() must be known

	// determine _y_size
	_ysize = 1;
	for(int i=0; i<_tags.size(); ++i)
		_ysize*=_tags[i].size();

	// determine _path_num from _order and _y_size
	_path_num=(int)pow((double)_ysize,_order+1);

	// prepare _y_marginal from _order and _y_size
	_y_marginal.resize(_order+1);
	for(int i=0;i<=_order;i++){
		int len=(int)pow(_ysize,i+1);
		_y_marginal[i]=vector<int>(len,0);
	}

	// prepare tag2int maps
	_tag2int.resize(_tags.size());
	for(int j=0; j<_tags.size(); ++j)
		for(int i=0; i<_tags[j].size(); ++i)
			_tag2int[j][_tags[j][i]]=i;

#ifdef SPARSE_W
	_sizes = vector <int> (1,_ysize);
	for(int x=1; x<=_order; x++)
		_sizes.push_back(_sizes[x-1]*_ysize);
#endif

	// build the forward and backward index map
	_feature2trellis.resize(_clique_types.size());
	_trellis2feature.resize(_clique_types.size(),vector<int>(_path_num));
	for(int x=0; x<_clique_types.size(); ++x){			// for each clique type
		for(int y=0; y<_path_num; ++y){					// for each cell in trellis
			vector <int> fids;
			int fid = trellisID2featureID(fids, y, _clique_types[x]);
			_trellis2feature[x][y] = fid;
			if(fid>=_feature2trellis[x].size())
				_feature2trellis[x].resize(fid+1);
			_feature2trellis[x][fid].push_back(y);
		}
	}
}

bool FM3N::generate_features(const string& training_file){
	char line[MAXSTRLEN];
	vector<char *>table;// table[i,j] = table[i*_cols+j]
	charlist table_str;
	table_str.set_size(PAGESIZE);//1 page memory

	// clear FeatureInfo->freq
	for(map<char *, FeatureInfo, str_cmp>::iterator it=_xindex.begin(); it!=_xindex.end(); ++it){
		it->second.freq=0;
		if(it->second.label_marginal)
			it->second.label_marginal->clear();
	}

	// clear _y_marginal
	for(int order=0; order<=_order; ++order)
		for(int y=0; y<_y_marginal[order].size(); ++y)
			_y_marginal[order][y]=0;

	fprintf(stderr,"Generating features ... (total # of weights, per 10000 lines)\n");
	int lines=0;
	istream &fin = *OpenRead(training_file);
	while(!fin.eof()){
		fin.getline(line,MAXSTRLEN-1);
		lines++;
		if(!(lines%10000)){
			fprintf(stderr,"%d.. ",_w_size);
			fflush(stderr);
		}
		if(line[0]){
			vector<char *>columns;
			split_string(line,"\t",columns,read_mode);
			for(int i=0;i<_cols;i++){
				char *p=table_str.push_back(columns[i]);
				table.push_back(p);
			}
		}else if(table.size()){
			add_x(table);
			table.clear();//prepare for new table
			table_str.clear();
		}
	}
	if(table.size()){//non-empty line
		add_x(table);
		table.clear();//prepare for new table
		table_str.clear();
	}
	CloseIO(&fin);
	fprintf(stderr,"%d Done!\n", _w_size);
	fprintf(stderr,"%d lines training data loaded\n", lines);

	return true;
}
void FM3N::reload_training(char *train_file){
	int old_w_size = _w_size;
	generate_features(train_file);
	if(old_w_size!=_w_size)
		cerr << "Warning: w_size has changed from " << old_w_size << " -> " << _w_size << endl;
}
int FM3N::get_index(vector<char *> &table, int start, int offset, string& comp_str, int comp_col, bool accept_equal, int ncol){
	assert(offset!=0);
	int acc=0, posi=start, rows=table.size()/ncol;
	if(offset>0){		// rightward
		if(offset>1){	// starting from next token
			offset--;
			posi++;
		}
		for(; posi<rows; posi++){
			if((accept_equal)?comp_str==table[posi*ncol+comp_col]:comp_str!=table[posi*ncol+comp_col])
				++acc;
			if(acc==offset)
				break;
		}
	}else{				// leftward
		if(offset<-1){	// starting from prev token
			offset++;
			posi--;
		}
		for(; posi>=0; posi--){
			if((accept_equal)?comp_str==table[posi*ncol+comp_col]:comp_str!=table[posi*ncol+comp_col])
				--acc;
			if(acc==offset)
				break;
		}
	}
	return posi;
}

int FM3N::make_feature_string(char *s, templet &pat, int i, int j, vector<char *> &table, int ncol)
{
	char ss[8][x_word_len];

	// strip header if no_header
	if(pat.mask&1)
		s[0]=0;
	else
		sprintf(s, "%d:", j);

	int index1,index2,k;
	int rows=table.size()/ncol;
	int x_size=pat.x.size();

	for(k=0;k<x_size;k++){
		if(pat.skip_cols.empty()){
			index1 = pat.x[k].first+i;
		}else{
			index1 = get_index(table, i, pat.x[k].first, pat.skip_words[k], pat.skip_cols[k]&0x7fffffff, (pat.skip_cols[k]&0x80000000)==0, ncol);
		}
		index2=pat.x[k].second;
		assert(index2>=0 && index2<ncol-1);
		char * s1 = ss[k];
		if(index1<0){
			index1=-index1-1;
			sprintf(s1,"B_%d",index1); //B_0 for example
		}else if(index1>=rows){
			index1-=rows;
			sprintf(s1,"E_%d",index1);	//E_0 for example
		}else if(!strlen(table[index1*ncol+index2])){
			s[0]=0;			//null feature
			return k;
		}else{
			strcpy(s1,table[index1*ncol+index2]);
		}
	}
	if(pat.func==1){
		if(x_size>=2){   // string comparison without string identity
			bool all_diff=true;
			for(int x=1; x<x_size; x++){
				if(strcmp(ss[x-1],ss[x]))
					strcat(s,"!=");
				else{
					strcat(s,"==");
					all_diff=false;
				}
			}
			if(all_diff)
				s[0]=0;
		}else
			s[0]=0;
	}else if(pat.func==2){
		if(x_size>=2){   // string comparison with string identity
			bool all_same=true;
			bool all_diff=true;
			for(int x=1; x<x_size; x++){
				if(strcmp(ss[x-1],ss[x])){
					strcat(s,"!=");
					all_same=false;
				}else{
					strcat(s,"==");
					all_diff=false;
				}
			}
			if(all_same)
				strcat(s,ss[0]);
			else if(all_diff)
				s[0]=0;
		}else
			s[0]=0;
	}else{
		if(pat.mask&2)
			qsort(&ss, x_size, x_word_len, &str_comp_func);

		for(k=0;k<x_size;k++){
			strcat(s,pat.words[k].c_str());
			strcat(s,"//");
			strcat(s,ss[k]);
			strcat(s,"//");
		}
	}
	return k;
}

inline int FM3N::get_composed_label(char**table_row, bool &bweighted){
	int key;
	char *label;
	bool weighted1;
	bweighted = false;
	label = get_pure_label(table_row[_cols-1], weighted1);
	if(weighted1) bweighted=true;
	if(!vector_search(_tags[0], label, key, str_cmp())){
		cerr << "Label not found: " << table_row[_cols-1] << endl;
		exit(1);
	}
	for(int x=1; x<=_y_max_id; ++x){
		label = get_pure_label(table_row[_cols-1-x], weighted1);
		if(weighted1) bweighted=true;
		int id;
		if(!vector_search(_tags[x], label, id, str_cmp())){
			cerr << "Label not found: " << table_row[_cols-1-x] << endl;
			exit(1);
		}
		key=key*_ysizes[x]+id;
	}
	return key;
}

inline void FM3N::get_decomposed_label(vector<int>&out_indices, int key){
	out_indices.resize(_y_max_id+1);
	for(int y=_y_max_id; y>=0; --y){
		out_indices[y]=key%_ysizes[y];
		key /= _ysizes[y];
	}
	assert(key==0);	// sanity check
}

inline int FM3N::hist2key(int *hist, int len){
	int key = 0;
	for(int x=0; x<len; x++)
		key = key*_ysize+hist[x];
	return key;
}

inline void FM3N::key2hist(int *hist, int key){
	for(int x=_order; x>=0; --x){
		hist[x] = key%_ysize;
		key /= _ysize;
	}
}

bool FM3N::add_x(vector<char *> &table){
	vector<int> y;
	int rows=table.size()/_cols;
	char s[1024];
	sequence seq;
	node* nod=_nodes.alloc(rows);
	seq.node_num=rows;
	seq.nodes=nod;

	_sequences.push_back(seq);
	y.resize(rows);
	for(int i=0;i<rows;i++){
		//char *lab = get_pure_label(table[(i+1)*_cols-1], nod[i].bWeighted);
		y[i]=get_composed_label(&table[i*_cols], nod[i].bWeighted);
		//vector_search(_tags, lab, y[i], str_cmp());		//get the tag of current node

		vector <int> keys(_gsize,0);
		{	// compute key at all orders
			int order;
			for(order=0; order<=_order; ++order){
				if(i-order<0)
					break;
				keys[order] = hist2key(&y[i-order],order+1);
				++_y_marginal[order][keys[order]];
			}
			nod[i].key=keys[order-1];
		}

		vector<clique*> clisp;	//features that affect on current nodes
		int cur_group=0;
		int n_clique_types = _clique_types.size();
		vector<vector<int> > feature_vectors(n_clique_types);
		for(int j=0;j<_templets.size();j++){
			// if first y's offset < 0, skip current feature???
			templet &pat=_templets[j];
			if(pat.y_max_history+i<0)
				continue;

			int k=make_feature_string(s, pat, i, j, table, _cols);

			if(s[0]){
				strcat(s,pat.words[k].c_str());

				// print feature string
				if(print_feature_file)
					(*print_feature_file) << s << ' ';

				//x obtained, insert x
				FeatureInfo *ppinfo;
				if(insert_x(s,&ppinfo,j)){
					int c = _feature2trellis[ppinfo->clique_type].size();
					_w_size += c;
				}

				//get clique
				feature_vectors[ppinfo->clique_type].push_back(ppinfo->global_id);

				// increment feature occurrence
				++(ppinfo->freq);

				// add label marginal for the current feature
				if(prune_method>=1)
					++(*ppinfo->label_marginal)[keys[ppinfo->clique_type]];
			}
		}

		for(int clique_type=0; clique_type<n_clique_types; ++clique_type){	//create new cliques
			clique cli;
			vector<node*> ns;
			vector <int> feat_ids;

			// push back nodes
			for(int k=_clique_hist_size[clique_type]; k<=0; k++)
				ns.push_back(nod+i+k);
			node ** np=_clique_node.push_back(&ns[0],ns.size());

			cli.nodes = np;
			cli.node_num = ns.size();
			cli.key = trellisID2featureID(feat_ids, keys[_order], _clique_types[clique_type]);		// get clique key

			vector<int> &feature_vector = feature_vectors[clique_type];
			if(feature_vector.size())
				cli.fvector=_clique_feature.push_back(&feature_vector[0],feature_vector.size());
			else
				cli.fvector=NULL;
			cli.feature_num=feature_vector.size();
			cli.groupid=clique_type;
			clique *new_clique=_cliques.push_back(&cli,1);
			clisp.push_back(new_clique);
		}

		//set node -> clique
		if(clisp.size())
			nod[i].cliques = _node_clique.push_back(&clisp[0],clisp.size());
		else
			nod[i].cliques = NULL;
		nod[i].clique_num =clisp.size();

		if(print_feature_file)
			(*print_feature_file) << endl;
	}
	if(print_feature_file)
		(*print_feature_file) << endl;
	return true;
}

bool FM3N::insert_x(char *target, FeatureInfo **ppinfo, int template_id){
	map<char *, FeatureInfo, str_cmp>::iterator p;
	p=_xindex.find(target);
	if(p!=_xindex.end()){	// existing feature
		*ppinfo = &p->second;
		return false;
	}else{					// new feature
		char *q=_x_str.push_back(target);
		int clique_type = _templets[template_id].clique_type;
		vector <int> *pmarginal = (prune_method>=1)?new vector<int>(_feature2trellis[clique_type].size(),0):NULL;
		FeatureInfo info={
				_w_size,		// global id
				0,				// freq
				template_id,	// template id
				clique_type,	// clique type
				pmarginal		// label_marginal
		};
		*ppinfo = &(_xindex.insert(make_pair(q,info)).first->second);
		return true;
	}
}

/*
void dump(map<char *, int, str_cmp> &str2int, vector<pair<int,int>> &x_freq_id){
	ofstream fout("z:\\tp.txt");
	for(auto it=str2int.begin(); it!=str2int.end(); ++it)
		fout << it->first << "\t" << x_freq_id[it->second].first << "\t" 
			<< x_freq_id[it->second].second << endl;
	fout.close();
	__asm int 3
}*/

void FM3N::count_feature_at_each_order(map<int,int> &ret){
	ret.clear();
	for(map<char *, FeatureInfo, str_cmp>::iterator it=_xindex.begin(); it!=_xindex.end(); ++it)
		++ret[it->second.clique_type];
}

string FM3N::ys2str(vector <pair<int,int> > &ys){
	ostringstream s;
	for(int x=0; x<ys.size(); ++x)
		s << "%y[" << ys[x].first << "," << ys[x].second << "]";
	return s.str();
}

void FM3N::shrink_feature(){
	prune_method=prune_params.empty()?0:prune_params[0];
	if(!_freq_thresh && prune_params.empty())
		return;

	fprintf(stderr,"Shrinking features ...\n");
    map <int, pair<int,int> > old2new;
    int old_x_index_size = _xindex.size();
	int old_lambda_size = _w_size;
    int new_lambda_size = 0;
	char temp[MAXSTRLEN];
	map <char*, FeatureInfo, str_cmp>::iterator it;
	map <int,int> num_feat_each_clique, orig_num_feat_each_clique;
	count_feature_at_each_order(orig_num_feat_each_clique);

	//dump(_xindex, _x_freq_id);
    if(prune_method>=1){	// INCOMPLETE YET
    	// compute y distribution for all clique orders
		vector <vector <double> > y_marginal (_order+1);	// no entry can be 0
		for(int order=0; order<=_order; ++order)
			norm_P(y_marginal[order], _y_marginal[order], 1.0);

		// compute KL divergence for all features
		vector <feature_entry> entries;
		for(it=_xindex.begin(); it!= _xindex.end(); ++it){
			char *feature_name = it->first;
			FeatureInfo &info = it->second;
			vector <int> &can_be_zero = *info.label_marginal;
			templet t = _templets[info.templet_id];
			//feature_entry e={feature_name, (!t.x.size()?1.0e10:KL_div(y_marginal[info.order],can_be_zero)*sum(can_be_zero))};	// default feature will NOT be prune away
			feature_entry e={feature_name, KL_div(y_marginal[info.clique_type],can_be_zero)*sum(can_be_zero)};		// default feature might be prune away
			entries.push_back(e);
		}

		std::sort (entries.begin(), entries.end());
		map<char *, FeatureInfo, str_cmp> new_xindex;

		// add selected features
		if(prune_method==1){			// no modification of feature order
			assert(prune_params.size()>=2);
			int param_limit = prune_params[1];
			for(int x=0; x<entries.size(); ++x){
				it = _xindex.find(entries[x].feature_name);
				assert(it!=_xindex.end());

				FeatureInfo info = it->second;
				old2new[info.global_id] = pair<int,int>(new_lambda_size,info.clique_type);
				info.global_id = new_lambda_size;
				info.label_marginal->clear();						// to save memory
				new_xindex[it->first] = info;
				new_lambda_size += _y_marginal[info.clique_type].size();
				if(new_lambda_size>=param_limit)
					break;
			}
		}else if(prune_method==2){		// modify feature order, according to KL*freq
			assert(prune_params.size()>=3);
			int param_limit = prune_params[1];
			int max_max_order_feats = prune_params[2];
			int num_max_order_feats = 0;
			int current_max_order = _order;
			double current_th=-1.0e10;
			for(int x=0; x<entries.size(); ++x){
				double score = entries[x].selector_value;
				it = _xindex.find(entries[x].feature_name);
				assert(it!=_xindex.end());

				FeatureInfo info = it->second;
				info.clique_type=min(info.clique_type,current_max_order);
				if(info.clique_type==_order){		// we are at max order
					if((++num_max_order_feats)>=max_max_order_feats && current_max_order>0){
						--current_max_order;
						current_th = score/_ysize;
					}
				}else{
					if(score<current_th && current_max_order>0){
						--current_max_order;
						current_th /= _ysize;
					}
				}
				old2new[info.global_id] = pair<int,int>(new_lambda_size,info.clique_type);
				info.global_id = new_lambda_size;
				info.label_marginal->clear();						// to save memory
				new_xindex[it->first] = info;
				new_lambda_size += _y_marginal[info.clique_type].size();
				if(new_lambda_size>=param_limit)
					break;
				if(new_lambda_size>old_lambda_size)
					new_lambda_size=0;
			}
		}else if(prune_method==3){		// modify feature order, according to KL*freq
			assert(prune_params.size()>=3);
			int param_limit = prune_params[1];
			int max_max_order_feats = prune_params[2];
			int num_max_order_feats = 0;
			int current_max_order = _order;
			double current_th=-1.0e10;
			for(int x=0; x<entries.size(); ++x){
				double score = entries[x].selector_value;
				it = _xindex.find(entries[x].feature_name);
				assert(it!=_xindex.end());

				FeatureInfo info = it->second;
				info.clique_type=min(info.clique_type,current_max_order);
				if(current_max_order==_order){		// we are at max order
					if((++num_max_order_feats)>=max_max_order_feats && current_max_order>0){
						--current_max_order;
						current_th = score/_ysize;
					}
				}else{
					if(score<current_th && current_max_order>0){
						--current_max_order;
						current_th /= _ysize;
					}
				}
				old2new[info.global_id] = pair<int,int>(new_lambda_size,info.clique_type);
				info.global_id = new_lambda_size;
				info.label_marginal->clear();						// to save memory
				new_xindex[it->first] = info;
				new_lambda_size += _y_marginal[info.clique_type].size();
				if(new_lambda_size>=param_limit)
					break;
				if(new_lambda_size>old_lambda_size)
					new_lambda_size=0;
			}
		}
		std::swap(new_xindex, _xindex);
    }else if(prune_params.size()>=2){		// frequency pruning, but limit parameters
    	int param_limit = prune_params[1];

		// compute and sort feature frequency for all features
		vector <feature_entry> entries;
		for(it=_xindex.begin(); it!= _xindex.end(); ++it){
			char *feature_name = it->first;
			FeatureInfo &info = it->second;
			feature_entry e={feature_name, -info.freq};
			entries.push_back(e);
		}
		std::sort (entries.begin(), entries.end());
		map<char *, FeatureInfo, str_cmp> new_xindex;

		// select new features
		for (int x = 0; x < entries.size(); ++x) {
			it = _xindex.find(entries[x].feature_name);
			assert(it != _xindex.end());

			FeatureInfo info = it->second;
			old2new[info.global_id] = pair<int, int>(new_lambda_size, info.clique_type);
			info.global_id = new_lambda_size;
			new_xindex[it->first] = info;
			new_lambda_size += _feature2trellis[info.clique_type].size();
			if (new_lambda_size >= param_limit)
				break;
		}
		std::swap(new_xindex, _xindex);
    }else{
    	for(it=_xindex.begin(); it!= _xindex.end();){		// default frequency pruning
			FeatureInfo &info = it->second;
			if (info.freq >= _templets[info.templet_id].freq_thresh)
			{
				int gram_num=_templets[info.templet_id].y.size();
				old2new[info.global_id] = pair<int,int>(new_lambda_size,info.clique_type);
				info.global_id = new_lambda_size;
				new_lambda_size += _feature2trellis[info.clique_type].size();
				++it;
			}else{
				_xindex.erase(it++);
			}
    	}
    }
	count_feature_at_each_order(num_feat_each_clique);

	_w_size=new_lambda_size;
	map<int, pair<int,int> >::iterator iter;
	freelist<int> temp_clique_feature;
	temp_clique_feature.set_size(PAGESIZE*16);
	_clique_feature.free();
	int n_clique_types=_clique_types.size();
	for(int i=0;i<_sequences.size();i++)			// for each sequence
	{
		sequence &seq=_sequences[i];
		for(int j=0;j<seq.node_num;j++)				// for each node
		{
			node &nod=seq.nodes[j];
			vector <vector<int> > newfs(n_clique_types);
			for(int k=0;k<nod.clique_num;k++)		// for each clique
			{
				if(!nod.cliques[k])
					continue;
				clique &cli=*nod.cliques[k];
				for(int ii=0;ii<cli.feature_num;ii++)	// for each feature
				{
					iter = old2new.find(cli.fvector[ii]);
					if(iter != old2new.end())
						newfs[iter->second.second].push_back(iter->second.first);
				}
			}
			for(int k=0; k<n_clique_types; ++k)
			{
				int *f;
				if(newfs[k].size())
					f=temp_clique_feature.push_back(&newfs[k][0],newfs[k].size());
				else
					f=NULL;
				assert(nod.cliques[k]);
				clique &cli=*nod.cliques[k];
				cli.fvector=f;
				cli.feature_num=newfs[k].size();
			}
		}
	}
	_clique_feature.clear();

	// copy feature vectors from temp_clique_feature to _clique_feature so that temp_clique_feature can be released
	for(int i=0;i<_sequences.size();i++)
	{
		sequence &seq=_sequences[i];
		for(int j=0;j<seq.node_num;j++)
		{
			node &nod=seq.nodes[j];
			for(int k=0;k<nod.clique_num;k++)
			{
				if(!nod.cliques[k])
					continue;
				clique &cli=*nod.cliques[k];
				if(cli.feature_num)
				{	
					int *f=_clique_feature.push_back(cli.fvector,cli.feature_num);
					cli.fvector=f;
				}else{
					cli.fvector=NULL;
				}
			}
		}
	}
	cerr<<"features shrinked "<<old_x_index_size<<" -> "<<_xindex.size()<<endl;
	cerr<<"parameters shrinked "<<old_lambda_size<<" -> "<<new_lambda_size<<endl;
	for(int clique_type=0; clique_type<n_clique_types; ++clique_type)
		cerr << "clique " << clique_type << " ( " << ys2str(_clique_types[clique_type]) << " ) : "
			<< orig_num_feat_each_clique[clique_type] << " -> " << num_feat_each_clique[clique_type] << endl;
    return;
}

void FM3N::initialize(){
	int i,j;
	//set key path for all nodes
	_node_anum=pow((double)_ysize,_order);
	_path_num=_node_anum*_ysize;
#ifndef SPARSE_W
	if(_kernel_type==LINEAR_KERNEL){
		if(!_w){//if _w != NULL => _w has been initalized with model file
			_w=new double[_w_size];
			memset(_w,0,sizeof(double)*_w_size);
		}
	}
#endif
	//initialize mu
	if(!_mu){
		//vector<double> mu;// marginal probabilities for all paths
		for(i=0;i<_sequences.size();i++){
			sequence &seq = _sequences[i];
			//seq.mu_index=mu.size();
			for(j=0;j<seq.node_num;j++){
				//int mu_size=mu.size();
				//mu.resize(mu_size+_path_num,0);
				seq.mu[j*_path_num+seq.nodes[j].key]=1;
			}
		}
		_mu_size=0;
		_mu=NULL;
		//memcpy(_mu,&mu[0],sizeof(double)*_mu_size);
	}else{
		//relearn: assign seq.mu_index
//		int mu_size=0;
//		for(i=0;i<_sequences.size();i++){
//			_sequences[i].mu_index=mu_size;
//			mu_size+=_sequences[i].node_num*_path_num;
//		}
	}
	_head_offset=-log((double)_ysize)*_order;
	_clique_kernel.resize(_gsize);
	_path_kernel.resize(_path_num*_path_num);
}

bool FM3N::find_violation(sequence &seq, double &kkt_violation){
	int i,j;
	kkt_violation=0;

//	forward_backward_viterbi(_ysize,_order,seq.node_num,_alpha_lattice,_optimum_alpha_lattice,_optimum_alpha_paths,_head_offset,LOGZERO,LOGZERO,INF);
//	forward_backward_viterbi(_ysize,_order,seq.node_num,_v_lattice,_optimum_v_lattice,_optimum_v_paths,0,0,-INF,INF);
	forward_backward_viterbi(_ysize,_order,seq.node_num,_alpha_lattice,_optimum_alpha_lattice,_optimum_alpha_alphaBetaPath[0],_optimum_alpha_alphaBetaPath[1],_head_offset,LOGZERO,LOGZERO,INF);
	forward_backward_viterbi(_ysize,_order,seq.node_num,_v_lattice,_optimum_v_lattice,_optimum_v_alphaBetaPath[0],_optimum_v_alphaBetaPath[1],0,0,-INF,INF);
	
	vector<int> *y1_ptr, *y2_ptr;
	vector<int> v_index(2);
	vector<double> vyc;
	vector<int> a_index(2);
	vector<double> ayc;
	double v_yc;
	int v_yc_index;
	double a_yc;
	int a_yc_index;
	int y1=-1;
	int y2;
	int violation=0;
	//check
	
	for(i=0;i<seq.node_num;i++){
		
		//order v,alpha
		//v_index.clear();
		vyc.clear();
		//a_index.clear();
		ayc.clear();
		vyc.assign(_optimum_v_lattice.begin()+i*_path_num,_optimum_v_lattice.begin()+(i+1)*_path_num);
		//merge_sort(vyc,v_index);
		find_top2(vyc,v_index);
		ayc.assign(_optimum_alpha_lattice.begin()+i*_path_num,_optimum_alpha_lattice.begin()+(i+1)*_path_num);
		//merge_sort(ayc,a_index);
		find_top2(ayc,a_index);

		for(j=0;j<_path_num;j++){
			//find v(~y_c),alpha(~y_c)

			if(v_index[0]!=j){
				v_yc=vyc[v_index[0]];
				v_yc_index=v_index[0];
			}else{
				v_yc=vyc[v_index[1]];
				v_yc_index=v_index[1];
			}

			if(a_index[0]!=j){
				a_yc=ayc[a_index[0]];
				a_yc_index=a_index[0];
			}else{
				a_yc=ayc[a_index[1]];
				a_yc_index=a_index[1];
			}

			if(ayc[j]==LOGZERO && vyc[j]>v_yc+EPS){			//violation=1;
				if(kkt_violation<vyc[j]-v_yc){
					kkt_violation=vyc[j]-v_yc;
					y1=i*_path_num+j;
					//assert(a_yc>LOGZERO);
					y2=i*_path_num+a_yc_index;
//					_path1=_optimum_v_paths[y1];
//					_path2=_optimum_alpha_paths[y2];
					y1_ptr=_optimum_v_alphaBetaPath;
					y2_ptr=_optimum_alpha_alphaBetaPath;
				}
			}else if(ayc[j]>LOGZERO && vyc[j]<v_yc-EPS){	//violation=2;
				if(kkt_violation<v_yc-vyc[j]){
					kkt_violation=v_yc-vyc[j];
					y1=i*_path_num+j;
					//assert(v_yc>vyc[j]);
					y2=i*_path_num+v_yc_index;
//					_path1=_optimum_alpha_paths[y1];
//					_path2=_optimum_v_paths[y2];
					y1_ptr=_optimum_alpha_alphaBetaPath;
					y2_ptr=_optimum_v_alphaBetaPath;
				}
			}
		}
	}
	if(y1==-1)
		return false;
	get_optimum_path(_path1, y1/_path_num, y1%_path_num, _ysize, seq.node_num, _node_anum, y1_ptr[0], y1_ptr[1]);
	get_optimum_path(_path2, y2/_path_num, y2%_path_num, _ysize, seq.node_num, _node_anum, y2_ptr[0], y2_ptr[1]);
	return true;
}
void FM3N::build_alpha_lattice(sequence &seq){
	//_alpha_lattice[i*_path_num+j] is the log distribution on the marginal probability
	int i,j;
	memset(&_alpha_lattice[0],0,seq.node_num*_path_num*sizeof(double));
	map <int,double> &mu = seq.mu;
	for(map<int,double>::iterator it=mu.begin(); it!=mu.end(); ++it)
		_alpha_lattice[it->first] = it->second;

	//memcpy(&_alpha_lattice[0],_mu+seq.mu_index,seq.node_num*_path_num*sizeof(double));
	vector<double> margin(_node_anum);//separator function 
	for(i=0;i<seq.node_num-1;i++){//i th node , j th path
		fill(margin.begin(),margin.end(),0);
		for(j=0;j<_path_num;j++)
			margin[j%_node_anum]+=_alpha_lattice[i*_path_num+j];
		for(j=0;j<_path_num;j++)//if margin>0, _alpha_lattice[i*_path_num+j]=log(_alpha_lattice[i*_path_num+j]/margin[j%_node_anum]), otherwise, _alpha_lattice[i*_path_num+j]=0
			if(_alpha_lattice[i*_path_num+j]>0)
				_alpha_lattice[i*_path_num+j]=log(_alpha_lattice[i*_path_num+j]/margin[j%_node_anum]);	//convert to log probability
			else if(margin[j%_node_anum]>0)
				_alpha_lattice[i*_path_num+j]=LOGZERO;//log(0)
			else
				_alpha_lattice[i*_path_num+j]=_head_offset;
	}
	for(j=0;j<_path_num;j++){//i=seq.node_num-1
			if(_alpha_lattice[i*_path_num+j]>0)
				_alpha_lattice[i*_path_num+j]=log(_alpha_lattice[i*_path_num+j]);
			else
				_alpha_lattice[i*_path_num+j]=LOGZERO;//log(0)
	}
}
void FM3N::build_v_lattice(sequence &seq){
	//build _v_lattice
	if(_model==TEST_MODEL && _v_lattice.size()<seq.node_num*_path_num)
		_v_lattice.resize(seq.node_num*_path_num);
	fill(_v_lattice.begin(),_v_lattice.end(),0);
	if(_kernel_type==LINEAR_KERNEL){
		//LEARN_MODEL: v(x,y)=w'f(x,y)+l(x,y)=\sum_t w'f(x,y,t)+l(x,y,t)=\sum_t phi(x,y,t)
		//TEST_MODEL:  v(x,y)=w'f(x,y)       =\sum_t w'f(x,y,t)
		for(int i=0;i<seq.node_num;i++){				// for each node
			//build phi(x,y,t) for each path
			node &n1=seq.nodes[i];
			double *v_lattice = &_v_lattice[i*_path_num];
			for(int j=0;j<n1.clique_num;j++){			// for each clique
				clique &cli=*(n1.cliques[j]);
				vector<vector<int> > &group=_feature2trellis[cli.groupid];
				for(int k=0;k<cli.feature_num;k++){		// for each feature
#ifdef	SPARSE_W
					int feature=cli.fvector[k], end=_xend[feature], start;
					// find the position >=featureID
					map<int,double>::iterator it=_w.find(feature);
					if(it==_w.end())
						it=_w.upper_bound(feature);
					while(it->first < end && it!=_w.end()){
						int ii = it->first - feature;
						vector<int> &groupii = group[ii];
						for(int jj=0; jj<groupii.size(); ++jj)
							v_lattice[groupii[jj]] += it->second;
						++it;
					}
#else
					double *w = &_w[cli.fvector[k]];
					for(int ii=0, II=group.size();ii<II;++ii)
						for(int jj=0, JJ=group[ii].size();jj<JJ;++jj)
							v_lattice[group[ii][jj]] += w[ii];
#endif
				}
			}
			if(_model==LEARN_MODEL){
				//assert(n1.factor==1 || n1.factor==2);
				for(int j=0;j<_path_num;j++){
					vector <int> ref, sys;
					get_decomposed_label(ref, n1.key%_ysize);
					get_decomposed_label(sys, j%_ysize);
					double loss=0;
					for(int j=0; j<=_y_max_id; ++j)
						if(ref[j]!=sys[j])
							loss+=_loss[j][ref[j]][sys[j]];
					if(n1.bWeighted)
						loss*=_B;
					if(loss!=0)
						v_lattice[j]+=loss;
/*					if(j%_ysize!=n1.key%_ysize){
						_v_lattice[i*_path_num+j]+=_loss[n1.key%_ysize][j%_ysize];
					}*/
				}
			}
		}
	}else{
		//v(x,y)=\sum_t phi(x,y,t)=\sum_t w'f(x,y,t)+l(x,y,t)=\sum_t C*\sum_{x`,y`} a(x`,y`)Df(x`,y`)f(x,y,t) + l(x,y,t)
		//phi(x,y,t)
		//=C*\sum_{x`}\{ \sum_{t`} [ K(x`,y_{x`},t`,x,y,t)-\sum_{y`_{t`}} (  K(x`,y`_{t`},x,y,t)mu(y`_{t`}) ) ] \}
		//  +I(y_t != {y_x}_t)
		//=C*\sum_{x`}\{ \sum_{t`} [ \sum_{q \in x_t, q` \in x`_t`}I({y`_x`_t`}_q`=={y_t}_q) K(q,q`)
		//						-mu(y`_{t`}\sum_{q \in x_t, q` \in x`_t`} I({y`_t`}_q`=={y_t}_q) K(q,q`) ] \} +I(y_t != {y_x}_t)
		//thus, need to calculate kernels K(q,q`) forall clique pair (q,q`)

		//K(x`,y`,t`,x,y,t)=\sum_{q \in x_t, q` \in x`_t`}I({y`_t`}_q`=={y_t}_q) K(q,q`)
		for(int i=0;i<seq.node_num;i++){
			//build phi(x,y,t) for each path
			node &n1=seq.nodes[i];
			//calculate n1's kernel
			for(int j=0;j<_sequences.size();j++){
				if(_print_level>1)
					cerr<<"\t\tkernel compute: k(seq["<<i<<"],seq["<<j<<"])"<<endl;
				for(int k=0;k<_sequences[j].node_num;k++){
					node &n2=_sequences[j].nodes[k];
					get_kernel(n1,n2);//calculate kernels
					//calculate v
					for(int ii=0;ii<_path_num;ii++){
						_v_lattice[i*_path_num+ii]+=_path_kernel[ii*_path_num+n2.key];
						for(int jj=0;jj<_path_num;jj++){
							_v_lattice[i*_path_num+ii]-=_path_kernel[ii*_path_num+jj]*_mu[_sequences[j].mu_index+k*_path_num+jj];
						}
					}
				}
			}
			for(int j=0;j<_path_num;j++){
				_v_lattice[i*_path_num+j]*=_C;
				if(_model==LEARN_MODEL)
					if(j%_ysize!=n1.key%_ysize)
						_v_lattice[i*_path_num+j]++;
			}
		}
	}
}
void FM3N::smo_optimize(sequence &seq){
	
	//calculate alpha1 , alpha2, v1 ,v2
	double alpha1=0;
	double alpha2=0;
	double v1=0;
	double v2=0;
	for(int i=0;i<seq.node_num;i++){
		alpha1+=_alpha_lattice[i*_path_num+_path1[i]];
		alpha2+=_alpha_lattice[i*_path_num+_path2[i]];
		v1+=_v_lattice[i*_path_num+_path1[i]];
		v2+=_v_lattice[i*_path_num+_path2[i]];
	}
	if(alpha1>LOGZERO)
		alpha1=exp(alpha1);
	else
		alpha1=0;
	if(alpha2>LOGZERO)
		alpha2=exp(alpha2);
	else
		alpha2=0;
	
	double kern=0;
	if(_kernel_type==LINEAR_KERNEL){		// faster: O(node_num) complexity
		//||f(x,_path1)-f(x,_path2)||_2^2
		// Step 1: build the 2 sparse vectors
		map <int,pair<int,int> > fvec_pair;
		for(int i=0;i<seq.node_num;i++){
			if(_path1[i]==_path2[i])
				continue;

			int J = seq.nodes[i].clique_num;
			clique **cs = seq.nodes[i].cliques;
			for(int j=0; j<J; ++j){
				int K=cs[j]->feature_num;
				int *fv=cs[j]->fvector;
				vector<int> &path2cliy = _trellis2feature[cs[j]->groupid];
				for(int k=0; k<K; ++k){
					++fvec_pair[fv[k]+path2cliy[_path1[i]]].first;
					++fvec_pair[fv[k]+path2cliy[_path2[i]]].second;
				}
			}
		}
		// Step 2: compute the norm of their difference vector
		for(map <int,pair<int,int> >::iterator it=fvec_pair.begin(); it!=fvec_pair.end(); ++it){
			int diff = it->second.first-it->second.second;
			kern += diff*diff;
		}
	}else{									// slower: O(node_num^2) complexity
		//||f(x,_path1)-f(x,_path2)||_2^2=\sum_{i,j} [K(_path1,i;_path1,j)+K(_path2,i;_path2,j)-K(_path1,i;_path2,j)-K(_path2,i;_path1,j)]
		int path_list_1[4];
		int path_list_2[4];
		for(int i=0;i<seq.node_num;i++){
			for(int j=0;j<seq.node_num;j++){
				path_list_1[0]=_path1[i];
				path_list_1[1]=_path2[i];
				path_list_1[2]=_path1[i];
				path_list_1[3]=_path2[i];

				path_list_2[0]=_path1[j];
				path_list_2[1]=_path2[j];
				path_list_2[2]=_path2[j];
				path_list_2[3]=_path1[j];

				get_kernel(seq.nodes[i],seq.nodes[j],path_list_1,path_list_2,4);
				kern+=_path_kernel[0];
				kern+=_path_kernel[1];
				kern-=_path_kernel[2];
				kern-=_path_kernel[3];
			}
		}
	}

	//delta=max{-alpha(_path1), min{alpha(_path2),delta_v/(_C*||f(x,_path1)-f(x,_path2)||_2^2)}}
	double delta=(v1-v2)/(_C*kern);
	delta=delta<alpha2?delta:alpha2;
	delta=delta>(-alpha1)?delta:(-alpha1);

	//update mu
	//for each mu on _path1
	//	mu+=delta
	//for each mu on _path2
	//	mu-=delta
	for(int i=0;i<seq.node_num;i++){
//		_mu[i*_path_num+_path1[i]+seq.mu_index]+=delta;
//		_mu[i*_path_num+_path2[i]+seq.mu_index]-=delta;
		seq.mu[i*_path_num+_path1[i]]+=delta;
		seq.mu[i*_path_num+_path2[i]]-=delta;
	}

	//update obj
	//obj=1/2 C||\sum_{X,Y} a_X(Y)Df(X,Y)||_2^2 - \sum_{X,Y} a_X(Y) Dt(X,Y)
	//v(x,y)= C*\sum_{x`,y`} a(x`,y`)Df(x`,y`)f(x,y) + l(x,y)
	//=>
	//D obj=-delta[v(x,y1)-v(x,y2)]+1/2 C delta^2 ||f(x,y1)-f(x,y2)||_2^2
	_obj+=-(v1-v2)*delta+0.5*_C*kern*delta*delta;
	if(_kernel_type==LINEAR_KERNEL){
		//update _w
		//_w=C*\sum_{x,y}a(x,y)Df(x,y)=C*\sum_{x,y,t}mu(x,y,t)Df(x,y,t)=C*\sum_{x,y}f(x,y_x)-\sum_{x,y,t}mu(x,y,t)f(x,y,t)
		//=>
		//D _w=-C*\sum_{x,y,t}Dmu(x,y,t)f(x,y,t)=-C*delta*[f(x,y1)+f(x,y2)]
		for(int i=0;i<seq.node_num;i++){					// for each node
			for(int j=0;j<seq.nodes[i].clique_num;j++){		// for each clique
				if(!seq.nodes[i].cliques[j])
					continue;
				clique &cli=*(seq.nodes[i].cliques[j]);
				for(int k=0;k<cli.feature_num;k++){			// for each feature
					_w[cli.fvector[k]+_trellis2feature[cli.groupid][_path1[i]]]-=delta*_C;
					_w[cli.fvector[k]+_trellis2feature[cli.groupid][_path2[i]]]+=delta*_C;
				}
			}
		}
	}
}


void FM3N::sort_feature(sequence &seq){
//for each node
//	sort clique by clique.group_id, this has been automatically set since cliques are generated in template order
//for each clique
//	sort fvector
	int i,j;

	for(i=0;i<seq.node_num;i++){
		for(j=0;j<seq.nodes[i].clique_num;j++){
			if(!seq.nodes[i].cliques[j]||!seq.nodes[i].cliques[j]->fvector)
				continue;
			qsort(seq.nodes[i].cliques[j]->fvector,seq.nodes[i].cliques[j]->feature_num,sizeof(int),int_cmp);
		}
	}
}
/*
void FM3N::assign_tag(sequence &seq, vector<int> &node_tag)
{
	int i,j,k;
	for(i=0;i<seq.node_num;i++){
		seq.nodes[i].key=0;
		for(j=0;j<=_order;j++){
			if(i+j>=_order)
				seq.nodes[i].key=seq.nodes[i].key*_ysize+node_tag[i+j-_order];
		}
	}
	for(i=0;i<seq.node_num;i++)
	{
		node &nod=seq.nodes[i];
		for(j=0;j<nod.clique_num;j++)
		{
			if(!nod.cliques[j])
				continue;
			clique &cli=*(nod.cliques[j]);
			int key=0;
			for(k=0;k<cli.node_num;k++)
				key= key*_ysize +cli.nodes[k]->key%_ysize;
			cli.key=key;
		}
	}
}
*/
void FM3N::node_margin(sequence &seq, vector<vector<double> >&node_p,vector<double> &alpha, vector<double> &beta, double &z)
{
	int i,j;
	node_p.resize(seq.node_num);
	for(i=0;i<seq.node_num;i++)
		node_p[i].resize(_ysize);
	vector<int> first_cal(_ysize*seq.node_num,1);
	for(i=0;i<seq.node_num;i++)
	{
		int *cur_first=&first_cal[_ysize*i];
		double *cur_p=&node_p[i][0];
		for(j=0;j<_node_anum;j++)
		{
			int index = j % _ysize;
			if(cur_first[index])
			{
				cur_first[index]=0;
				cur_p[index]=alpha[i*_node_anum+j]+beta[i*_node_anum+j];
			}else
				cur_p[index]=log_sum_exp(alpha[i*_node_anum+j]+beta[i*_node_anum+j],cur_p[index]);
		}
		for(j=0;j<_ysize;j++){
			if(_margin==1)
				cur_p[j]=exp(cur_p[j]-z);
			else if(_margin==2)
				cur_p[j]=cur_p[j]-z;
		}
	}
}
double FM3N::path_cost(sequence &seq, vector<double>& lattice){
	int i;
	double c=0;
	for(i=0;i<seq.node_num;i++)
		c+=lattice[_path_num*i+seq.nodes[i].key];
	return c;
}
vector <double> FM3N::compute_hypo_score(vector<char*> &table, vector<vector<char*> > &tags, int ncol){
	vector <double> costs;
	/* TODO
	sequence seq;
	_model=TEST_MODEL;
	generate_sequence(table,seq,ncol);
	sort_feature(seq);
	build_v_lattice(seq);

	// assign node keys
	bool dummy;
	for(int j=0; j<tags.size(); j++){
		vector <int> y;
		vector <char*> &tag = tags[j];
		for(int i=0;i<seq.node_num;i++){
			y.push_back(0);
			char *lab = get_pure_label(tag[i], dummy);
			if(!vector_search(_tags, lab, y.back(), str_cmp())){	//get the tag of current node
				cerr << "Label not found: " << lab << endl;
				exit(1);
			}

			int key;
			for(int order=0; order<=_order; ++order){
				if(i-order<0)
					break;
				key = calc_key(&y[i-order],order+1);
			}
			seq.nodes[i].key=key;
		}
		double cost = path_cost(seq, _v_lattice);
		costs.push_back(cost);
	}*/
	return costs;
}
void FM3N::tag(sequence &seq, vector<vector<int> > &best_tag_indices){	// from v-lattice
	int model=_model;
	_model = TEST_MODEL;
	build_v_lattice(seq);
	_model = model;

	vector<vector<int> > best_path;
	viterbi(seq.node_num, _order, _ysize, _nbest, _v_lattice, best_path);
	int i,j;
	for(i=0;i<best_path.size();i++){
		vector<int> cur_tag(seq.node_num);
		for(j=0;j<seq.node_num;j++){
			cur_tag[j]=best_path[i][j];
		}
		best_tag_indices.push_back(cur_tag);
	}
}
void FM3N::tag(vector<char*> &table, vector<string> &best_tag, vector<vector<double> > &marginal, int ncol){	// compute posterior
	/* TODO
	double z;
	vector<double> alpha, beta;
	vector<vector<int> > best_path;
	sequence seq;

	_model=TEST_MODEL;
	generate_sequence(table,seq,ncol);
	sort_feature(seq);
	build_v_lattice(seq);

	viterbi(seq.node_num, _order, _ysize, _nbest, _v_lattice, best_path);
	best_tag.clear();
	for(int j=0;j<seq.node_num;j++)
		best_tag.push_back(_tags[best_path.back()[j]]);

	forward_backward(_ysize, _order, seq.node_num, _v_lattice, alpha, beta, z);
	node_margin(seq, marginal, alpha, beta, z);*/
}
void FM3N::tag(vector<char*> &table, vector<vector<vector<string> > > &best_tag, int ncol){	// output n-best or 1-best
	_model=TEST_MODEL;
	sequence seq;
	generate_sequence(table,seq,ncol);
	sort_feature(seq);
	build_v_lattice(seq);
	vector<vector<int> > best_path;
	int rows=table.size()/ncol;
	vector<vector<int> > valid_v(rows);
	
//	if(_nbest==1){
	viterbi(seq.node_num, _order, _ysize, _nbest, _v_lattice, best_path);
	for(int i=0;i<best_path.size();i++){
		vector<vector<string> > cur_tag(seq.node_num);
		for(int j=0;j<seq.node_num;j++){
			vector <int> ids;
			get_decomposed_label(ids,best_path[i][j]);
			for(int k=0; k<=_y_max_id; ++k)
				cur_tag[j].push_back(_tags[k][ids[k]]);
		}
		best_tag.push_back(cur_tag);
	}
}
void FM3N::generate_sequence(std::vector<char*> &table, sequence &seq, int ncol){
	int rows=table.size()/ncol;
	char s[1024];
	node* nod=_test_nodes.alloc(rows);
	seq.node_num=rows;
	seq.nodes=nod;
	int n_clique_types = _clique_types.size();
	for(int i=0;i<rows;i++)
	{
		nod[i].key=0;				//random initialize
		vector<clique*> clisp;		//features that affect on current nodes
		int cur_group=0;
		vector<vector<int> > feature_vectors(n_clique_types);
		for(int j=0;j<_templets.size();j++)
		{
			//get first y's offset
			templet &pat=_templets[j];
			if(pat.y_max_history+i<0)
				continue;

			int k=make_feature_string(s, pat, i, j, table, ncol);

			if(s[0]){
				strcat(s,pat.words[k].c_str());
				//x obtained, insert x
				map<char *, FeatureInfo, str_cmp>::iterator it;
				it=_xindex.find(s);
				if(it!=_xindex.end())
					feature_vectors[it->second.clique_type].push_back(it->second.global_id);
			}
		}
		//RESUME
		for(int order=0; order<n_clique_types; ++order){
			clique cli;
			vector<node*> ns;
			for(int k=0;k<=order;k++)
				ns.push_back(nod+i+k-order);
			node ** np=_test_clique_node.push_back(&ns[0],ns.size());
			cli.nodes=np;
			cli.node_num=ns.size();

			vector<int> &feature_vector = feature_vectors[order];
			if(feature_vector.size()){
				cli.fvector=_test_clique_feature.push_back(&feature_vector[0],feature_vector.size());
			}else{
				cli.fvector=NULL;
			}
			cli.feature_num=feature_vector.size();
			cli.groupid=order;
			cli.key=0;//random initialize
			clique *new_clique=_test_cliques.push_back(&cli,1);
			clisp.push_back(new_clique);
		}
		//set node -> clique
		if(clisp.size())
			nod[i].cliques = _test_node_clique.push_back(&clisp[0],clisp.size());
		else
			nod[i].cliques = NULL;
		nod[i].clique_num =clisp.size();
	}
	_test_nodes.free();
	_test_node_clique.free();
	_test_cliques.free();
	_test_clique_node.free();
	_test_clique_feature.free();
}

bool FM3N::load_model(const string& model_file){
	int i,j,k,ii;
	char line[MAXSTRLEN];
	istream *pin = OpenRead(model_file);
	istream &fin = *pin;
	if(!fin.good()){
		cerr<<"model file: "<<model_file<<" not found"<<endl;
		return false;
	}
	//check version
	fin.getline(line,MAXSTRLEN-1);
	char *p=strstr(line,"\t");
	_version=atoi(p);
	fprintf(stderr,"model version: 0.%d\n",_version);
	//load kernel type, kernel parameters
	fin.getline(line,MAXSTRLEN-1);
	_kernel_type=atoi(line);
	fin.getline(line,MAXSTRLEN-1);
	_kernel_s=atof(line);
	fin.getline(line,MAXSTRLEN-1);
	_kernel_d=atof(line);
	fin.getline(line,MAXSTRLEN-1);
	_kernel_r=atof(line);

	//load eta
	fin.getline(line,MAXSTRLEN-1);
	_eta=atof(line);
	//load C
	fin.getline(line,MAXSTRLEN-1);
	_C=atof(line);
	//load freq_thresh
	fin.getline(line,MAXSTRLEN-1);
	_freq_thresh=atoi(line);

	//load templates
	while(!fin.eof()){
		fin.getline(line,MAXSTRLEN-1);
		if(!add_templet(line))
			break;
	}
	_gsize=_order+1;
	_feature2trellis.resize(_gsize);
	cerr<<"template number: "<<_templets.size()<<endl;

	//get ysize
	_ysize=1;
	fin.getline(line,MAXSTRLEN-1);
	_y_max_id=atoi(line);
	_tags.clear();
	_ysizes.clear();
	for(int x=0; x<=_y_max_id; ++x){
		fin.getline(line,MAXSTRLEN-1);
		int ysize = atoi(line);
		_ysizes.push_back(ysize);
		_tags.push_back(vector<char*>());
		for(i=0;i<ysize;i++){
			fin.getline(line,MAXSTRLEN-1);
			char *q=_tag_str.push_back(line);
			_tags.back().push_back(q);
		}
		_ysize*=ysize;
		cerr << "tag " << x << " number: " << ysize << endl;
	}

	cerr<<"total tags number: "<<_ysize<<endl;
	set_group();
	cerr<<"model order: "<<_gsize<<endl;

	//get cols
	fin.getline(line,MAXSTRLEN-1);
	fin.getline(line,MAXSTRLEN-1);
	_cols=atoi(line);

	//load x
	fin.getline(line,MAXSTRLEN-1);
	fin.getline(line,MAXSTRLEN-1);
	int x_num=atoi(line);
	_xindex.clear();
#ifdef SPARSE_W
	_xend.clear();
#endif
	for(i=0;i<x_num;i++){
		fin.getline(line,MAXSTRLEN-1);
		vector<char *> columns;
		split_string(line,"\t",columns);
		char *q=_x_str.push_back(columns[0]);
		FeatureInfo info;
		info.global_id = atoi(columns[1]);
		info.clique_type = atoi(columns[2]);
		info.label_marginal = NULL;
		if(columns.size()>=5){	// optionally load feature occurrence frequency and templet_id
			info.freq = atoi(columns[3]);
			info.templet_id = atoi(columns[4]);
		}
		_xindex[q]=info;
#ifdef SPARSE_W
		_xend[info.global_id] = info.global_id+_sizes[info.clique_type];
#endif
	}

	fin.getline(line,MAXSTRLEN-1);
	fin.getline(line,MAXSTRLEN-1);
	if(_kernel_type==LINEAR_KERNEL){
		//load obj
		_obj=atof(line);
		fin.getline(line,MAXSTRLEN-1);
		//load _w
		_w_size=atoi(line);
#ifdef SPARSE_W
		_w.clear();
#else
		_w=new double[_w_size];
#endif
		for(i=0;i<_w_size;i++){
			fin.getline(line,MAXSTRLEN-1);
#ifdef SPARSE_W
			int index;
			double val;
			sscanf(line,"%d%lg", &index, &val);
			_w[index] = val;
#else
			_w[i]=atof(line);
#endif
		}
		fin.getline(line,MAXSTRLEN-1);
		//load _mu
		fin.getline(line,MAXSTRLEN-1);
		_mu_size=atoi(line);
		_mu=new double[_mu_size];
		for(i=0;i<_mu_size;i++){
			fin.getline(line,MAXSTRLEN-1);
			_mu[i]=atof(line);
		}
		cerr<<_w_size<<" parameters loaded"<<endl;
		_get_kernel=&FM3N::linear_kernel;
		_get_kernel_list=&FM3N::linear_kernel;
	}else{
		//load _sequences info
		int seq_size=atoi(line);
		_sequences.resize(seq_size);
		for(i=0;i<_sequences.size();i++){
			sequence &seq=_sequences[i];
			
			fin.getline(line,MAXSTRLEN-1);
			sscanf(line,"%d\t%d",&seq.node_num,&seq.mu_index);
			node *nods=_nodes.alloc(seq.node_num);
			seq.nodes=nods;
			for(j=0;j<seq.node_num;j++){
				node &nod=nods[j];
				fin.getline(line,MAXSTRLEN-1);
				sscanf(line,"%d\t%d",&nod.key,&nod.clique_num);
				vector<clique*> clisp;
				for(k=0;k<nod.clique_num;k++){
					clique cli;
					fin.getline(line,MAXSTRLEN-1);
					sscanf(line,"%d\t%d\t%d\t%d",&cli.groupid,&cli.feature_num,&cli.node_num,&cli.key);
					vector<int> fvector(cli.feature_num);
					fin.getline(line,MAXSTRLEN-1);
					vector<char *> row;
					split_string(line,"\t",row);
					for(ii=0;ii<fvector.size();ii++){
						fvector[ii]=atoi(row[ii]);
					}
					if(fvector.size())
						cli.fvector=_clique_feature.push_back(&fvector[0],fvector.size());
					else
						cli.fvector=NULL;
					clique *new_clique=_cliques.push_back(&cli,1);
					clisp.push_back(new_clique);
				}
				if(clisp.size())
					nod.cliques=_node_clique.push_back(&clisp[0],clisp.size());
				else
					nod.cliques=NULL;
			}
		}
		//load _mu
		fin.getline(line,MAXSTRLEN-1);
		fin.getline(line,MAXSTRLEN-1);
		_obj=atof(line);
		fin.getline(line,MAXSTRLEN-1);
		_mu_size=atoi(line);
		_mu=new double[_mu_size];
		for(i=0;i<_mu_size;i++){
			fin.getline(line,MAXSTRLEN-1);
			_mu[i]=atof(line);
		}
		cerr<<_mu_size<<" parameters loaded"<<endl;

		if(_kernel_type==POLY_KERNEL){
			_get_kernel=&FM3N::poly_kernel;
			_get_kernel_list=&FM3N::poly_kernel;
		}else if(_kernel_type==NEURAL_KERNEL){
			_get_kernel=&FM3N::neural_kernel;
			_get_kernel_list=&FM3N::neural_kernel;
		}else if(_kernel_type==RBF_KERNEL){
			_get_kernel=&FM3N::rbf_kernel;
			_get_kernel_list=&FM3N::rbf_kernel;
		}
		_clique_kernel.resize(_gsize);
		_path_kernel.resize(_path_num*_path_num);
	}
	_path_num=pow((double)_ysize,_order+1);
	_node_anum=pow((double)_ysize,_order);//alpha(beta) number of each node
	_head_offset=-log((double)_ysize)*_order;
	switch(_kernel_type){
		case LINEAR_KERNEL: cerr<<"linear kernel: k(a,b)=<a,b>"<<endl;break;
		case POLY_KERNEL: cerr<<"polynomial kernel: k(a,b)=("<<_kernel_s<<"*<a,b>+"<<_kernel_r<<")^"<<_kernel_d<<endl;break;
		case RBF_KERNEL: cerr<<"rbf kernel: k(a,b)=exp{-"<<_kernel_s<<"*||a-b||^2}"<<endl;break;
		case NEURAL_KERNEL: cerr<<"neural kernel: k(a,b)=tanh("<<_kernel_s<<"*<a,b>+"<<_kernel_r<<")"<<endl;break;
	}
	return true;
}

void FM3N::print(ostream &fout){
	// build
	map<char *, FeatureInfo, str_cmp>::iterator it;
	vector <bool> bits(_w_size);
	vector <int> n_weights(_w_size);
	multimap <double, char*> g_map;
	cerr << "preparing 1..." << endl;
	for(it=_xindex.begin(); it!=_xindex.end(); ++it)
		bits[it->second.global_id] = true;

	cerr << "preparing 2..." << endl;
	for(it=_xindex.begin(); it!=_xindex.end(); ++it){
		char *feature_str = it->first;
		int feature_id = it->second.global_id;
		int x;
		for(x=feature_id+1; x<bits.size() && !bits[x]; x++);
		n_weights[feature_id] = x-feature_id;
#ifdef SPARSE_W
		double val = 1.0;
#else
		double val = std_deviation(&_w[feature_id], n_weights[feature_id]);
#endif
		g_map.insert(make_pair(val,feature_str));
	}

	// write
	cerr << "writing 1... _tags.size()="<< _tags.size() << endl;
	for(int y=0; y<=_y_max_id; ++y){
		fout << "tag " << y << " : ";
		for(int x=0; x<_tags[y].size(); x++)
			fout << _tags[y][x] << " ";
		fout << endl;
	}

	cerr << "writing 2... y_marginals" << endl;
	fout << "y-marginal-count: max-order=" << _order+1;
	for(int order=0; order<=_y_marginal.size(); ++order){
		fout << "order=" << order;
		for(int x=0; x<_y_marginal[order].size(); ++x)
			fout << ' ' << _y_marginal[order][x];
		fout << endl;
	}

	cerr << "writing 3... _xindex.size()=" <<_xindex.size() << endl;
	multimap <double, char*>::iterator mit;
	int n=0, N=g_map.size();
	for(mit=g_map.begin(); mit!=g_map.end(); ++mit,++n){
		char *feature_name = mit->second;
		fout << feature_name << " std=" << mit->first;
		fout << " rank=" << (N-n) << "/" << N;
		it = _xindex.find(feature_name);
		assert(it!=_xindex.end());
		FeatureInfo &info = it->second;
		fout << " order=" <<  info.clique_type;
		if(info.freq){
			fout << " freq=" << info.freq << " tid=" << info.templet_id;
		}
		if(sum(*(info.label_marginal))){
			fout << " label-marginal=";
			for(int x=0; x<_ysize; x++){
				fout << ' ' << (*info.label_marginal)[x];
			}
		}
		fout << endl;
		fout << std::scientific;
		for(int x=0,X=n_weights[info.global_id]; x<X; x++){
			fout << _w[info.global_id+x] << " ";
		}
		fout << endl;
	}
}

vector <int> & sequence::get_keys(vector <int> &keys, int _ysize){
	keys.clear();
	for(int x=0; x<node_num; x++)
		keys.push_back(nodes[x].key%_ysize);
	return keys;
}

