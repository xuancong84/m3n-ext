#include <fstream>
#include <iostream>
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

M3N::M3N(){
	read_mode=0;
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
	_get_kernel=&M3N::linear_kernel;
	_get_kernel_list=&M3N::linear_kernel;
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
	_w=NULL;
	_w_size=0;
	_max_iter=10;
	_obj=0;
	_print_level=0;
}

M3N::~M3N(){
	if(_mu){
		delete [] _mu;
		_mu=NULL;
	}
	if(_w){
		delete [] _w;
		_w=NULL;
	}
}

bool M3N::set_para(char *para_name, char *para_value){
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
			_get_kernel=&M3N::linear_kernel;
			_get_kernel_list=&M3N::linear_kernel;
		}else if(_kernel_type==POLY_KERNEL){
			_get_kernel=&M3N::poly_kernel;
			_get_kernel_list=&M3N::poly_kernel;
		}else if(_kernel_type==NEURAL_KERNEL){
			_get_kernel=&M3N::neural_kernel;
			_get_kernel_list=&M3N::neural_kernel;
		}else if(_kernel_type==RBF_KERNEL){
			_get_kernel=&M3N::rbf_kernel;
			_get_kernel_list=&M3N::rbf_kernel;
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
bool M3N::load_loss(string &firstline, istream &ifs){
	return true;
}
bool M3N::load_loss(const string& loss_file){
	if(loss_file.empty()){
		_loss.clear();
		for(int x=0; x<_ysize; x++){
			_loss.push_back(vector<double>());
			for(int y=0; y<_ysize; y++)
				_loss.back().push_back(x==y?0:1);
		}
		return true;
	}
	istream &ifs=*OpenRead(loss_file);
	if(!ifs.good()){
		cerr<<"can not open file "<<loss_file<<endl;
		return false;
	}
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
	_loss.clear();
	_loss.resize(_ysize, vector<double>(_ysize));
	for(int x=0; x<_ysize; ++x)
		for(int y=0; y<_ysize; ++y)
			_loss[x][y] = loss[labs2int[_tags[x]]][labs2int[_tags[y]]];

	cerr << "loss-type: " << _loss_type << endl;
	cerr << "loss-matrix: " << endl;
	for(int x=0; x<_loss.size(); x++){
		cerr << _tags[x];
		for(int y=0; y<_loss[x].size(); y++)
			cerr << '\t' << _loss[x][y];
		cerr << endl;
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
int M3N::calc_key(int *p, int len){
	int key = 0;
	for(int x=0; x<len; x++)
		key = key*_ysize+p[x];
	return key;
}
bool M3N::learn(const string& templet_file, const string& training_file,
		const string& model_file, const string& loss_file, bool relearn){
	cerr<<"pocket M3N"<<endl<<"version 0."<<_version<<endl<<"Copyright(c)2008 Media Computing and Web Intelligence LAB, Fudan Univ.\nAll rights reserved"<<endl;
	int i;
	_model=LEARN_MODEL;
	if(relearn){
		if(!load_model(model_file))
			return false;
		_templets.clear();
		_templet_group.clear();
		_path2cliy.clear();
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
	shrink_feature();
	for(i=0;i<_sequences.size();i++)
		sort_feature(_sequences[i]);
	vector<sequence>(_sequences).swap(_sequences);

	//write model part 1
	initialize();
	write_model(model_file,true);

	_tags.clear();
	_tag_str.clear();
	_xindex.clear();
	_x_str.clear();
	_templets.clear();
	
	cerr<<"sequence number: "<<_sequences.size()<<endl<<"feature number: "<<_w_size<<endl<<"C: "<<_C<<endl<<"B: "<<_B<<endl<<"freq_thresh: "<<_freq_thresh<<endl<<"eta: "<<_eta<<endl<<"max_iter: "<<_max_iter<<endl;
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
	_optimum_alpha_paths.resize(max_seq_len*_path_num);
	for(i=0;i<_optimum_alpha_paths.size();i++)
		_optimum_alpha_paths[i].resize(max_seq_len);
	_optimum_v_lattice.resize(max_seq_len*_path_num);
	_optimum_v_paths.resize(max_seq_len*_path_num);
	for(i=0;i<_optimum_v_paths.size();i++)
		_optimum_v_paths[i].resize(max_seq_len);
	int iter=0;
	double kkt_violation;
	double diff=1;
	int converge=0;
	clock_t start_time=clock();

	int C, P, R;
	double N;
	vector<vector<double> > _loss_orig = _loss;

	for(iter=0;iter<_max_iter;){

		if(!_loss_type.empty()){
			// compute total C,P,R and confusion table
			map <int, map<int, int> > conf_table;
			N=C=P=R=0;
			for(i=0;i<_sequences.size();i++){
				vector <vector<int> > labs;
				vector <int> refs;
				tag(_sequences[i], labs);
				_sequences[i].get_keys(refs,_ysize);
				computeCPR(refs, labs[0], C, P, R);
				confusion_table(conf_table, refs, labs[0]);
				N+=_sequences[i].node_num;
			}

			// adjust loss matrix
			_loss = _loss_orig;
			N=P;
			if(C && P && R){
				int converge;
				double old_F1 = 2.0*C/(R+P);
				for(int x=0; x<_loss_orig.size(); x++)
					for(int y=0; y<_loss_orig.size(); y++){
						int C1=0, P1=0, R1=0, C2=0, P2=0, R2=0;
						double _C=C, _P=P, _R=R;
						computeCPR(x,y,C1,P1,R1);
						computeCPR(x,x,C2,P2,R2);
						_C+=(C1-C2);
						_P+=(P1-P2);
						_R+=(R1-R2);
						double new_F1 = 2.0*_C/(_R+_P);
						_loss[x][y] = old_F1-new_F1;
					}
				
				if(_loss_type=="F1b"){
					// find smallest abs
					double sum=0, sum_orig=0;
					for(int i=0;i<4;i++){
						sum_orig += conf_table[i][4]*_loss_orig[i][4];
						sum_orig += conf_table[4][i]*_loss_orig[4][i];
						sum += conf_table[i][4]*_loss[i][4];
						sum += conf_table[4][i]*_loss[4][i];
					}
					if(sum==0 || sum_orig==0){
						fprintf(stderr,"error=0 converged\n");
						break;
					}
					double r = sum_orig/sum;
					vector<vector<double> > _loss2 = _loss;
					_loss = _loss_orig;
					for(int i=0;i<4;i++){
						_loss[i][4]=r*_loss2[i][4];
						_loss[4][i]=r*_loss2[4][i];
					}
				}
				/*
				converge = normalize_confusion_table(conf_table, _loss_orig, _loss);
				if(converge){
					printf("converged: normalize_confusion_table=%d\n", converge);
					break;
				}*/
			}
			cerr << "Using loss matrix:" << endl;
			for(int x=0; x<_loss.size(); x++){
				for(int y=0; y<_loss[x].size(); y++)
					cerr << '\t' << _loss[x][y];
				cerr << endl;
			}
		}

		//pass through sequences
		double old_obj=_obj;
		for(i=0;i<_sequences.size();i++){
			build_alpha_lattice(_sequences[i]);
			//view_lattice(_alpha_lattice, _sequences[i].node_num, _node_anum);
			build_v_lattice(_sequences[i]);
			//view_lattice(_v_lattice, _sequences[i].node_num, _node_anum);
			if(find_violation(_sequences[i],kkt_violation))
				smo_optimize(_sequences[i]);
			if(_print_level>0)
				fprintf(stderr,"\tseq: %d kkt_violation: %lf\n",i,kkt_violation);
		}
		
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
bool M3N::write_model(const string &model_file, bool first_part){
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
		fout<<_ysize<<endl;
		for(i=0;i<_tags.size();i++)
			fout<<_tags[i]<<endl;
		fout<<endl;

		//write x
		fout<<_cols<<endl<<endl;
		fout<<_xindex.size()<<endl;
		map<char*, FeatureInfo, str_cmp>::iterator it;
		for(it = _xindex.begin(); it != _xindex.end(); it++){
			//fout<<it->first<<'\t'<<it->second<<endl;
			FeatureInfo &info = it->second;
			fout<<it->first<<'\t'<<info.global_id<<'\t'<<info.order<<'\t'<<info.freq<<'\t'<<info.templet_id<<endl;
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
			fout<<_w_size<<endl;
			for(i=0;i<_w_size;i++){
				fout<<_w[i]<<endl;
			}
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


bool M3N::add_templet(char *line){
	if(!line[0]||line[0]=='#')	// skip comments
		return false;

	templet n;
	n.func=0;
	n.mask=0;
	n.full_str=line;
	n.freq_thresh=_freq_thresh;

	// now support multiple parts
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

	p=q-3;
	while(p=catch_string(p,"%y[","]",index_str)){
		index1=atoi(index_str);
		n.y.push_back(index1);
	}
	if(-n.y[0]>_order)
		_order=-n.y[0];
	_templets.push_back(n);
	return true;
}

bool M3N::load_templet(const string& templet_file){
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
	_templet_group.resize(_gsize);
	
	if(_order<0)// no _templets
		return false;
	return true;
}

bool M3N::check_training(const string& training_file){
	istream &fin=*OpenRead(training_file);
	if(!fin.good())	return false;
	char line[MAXSTRLEN];
	int lines=0;
	_cols=0;
	fprintf(stderr,"Check training ...(per 10000 lines)\n");
	while(!fin.eof()){//fgets(line,MAXSTRLEN-1,fp))
		fin.getline(line,MAXSTRLEN-1);
		lines++;
		if(!line[0]) continue;
		vector<char *>columns;
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

		int index;
		char *t=columns.back();//tag
		if(*t==weighted_node_sym)
			++t;
		if(!vector_search(_tags,t,index,str_cmp())){
			char *p=_tag_str.push_back(t);//copy string
			vector_insert(_tags,p,index);
		}

		if(!(lines%10000)){
			fprintf(stderr,".");
			fflush(stderr);
		}
	}
	CloseIO(&fin);
	_ysize=_tags.size();
	set_group();
	fprintf(stderr,"Done!\nNumber of labels = %d\n",_ysize);
	return true;
}

void M3N::set_group(){
	//calculate _templet_group, set the size of each group
	//_order and _tags.size() must be known

	for(int i=0; i<_tags.size(); ++i)
		_tag2int[_tags[i]]=i;

	// set for all possible clique order
	_templet_group.resize(_order+1);
	_y_marginal.resize(_order+1);
	for(int i=0;i<=_order;i++){
		int len=(int)pow(_ysize,i+1);
		_templet_group[i].resize(len);//group j has n offsets
		_y_marginal[i]=vector<int>(len,0);
	}

	// build the forward index map
	vector<int> path_index(_order+1,0);
	_path_num=(int)pow((double)_ysize,_order+1);
	for(int i=0;i<_path_num;i++){	//assosiate path i with _templet_group
		for(int j=0;j<_templet_group.size();j++){
			int offset=0;
			for(int k=0;k<j+1;k++)
				offset=offset*_ysize+path_index[j-k];
			_templet_group[j][offset].push_back(i);	//path i added to current group's offset
		}
		int j;
		for(j=0;j<_order+1 && path_index[j]==_ysize-1;j++);
		if(j==_order+1) break;
		path_index[j]++;
		for(j--;j>=0;j--)	path_index[j]=0;
	}

	// build the backward index map
	_path2cliy.resize(_templet_group.size());
	for(int i=0;i<_path2cliy.size();i++)
		_path2cliy[i].resize(_path_num,-1);
	for(int i=0;i<_templet_group.size();i++){
		for(int j=0;j<_templet_group[i].size();j++)
			for(int k=0;k<_templet_group[i][j].size();k++)
				_path2cliy[i][_templet_group[i][j][k]]=j;
	}
}

bool M3N::generate_features(const string& training_file){
	char line[MAXSTRLEN];
	vector<char *>table;// table[i,j] = table[i*_cols+j]
	charlist table_str;
	table_str.set_size(PAGESIZE);//1 page memory

	// clear FeatureInfo->freq
	for(map<char *, FeatureInfo, str_cmp>::iterator it=_xindex.begin(); it!=_xindex.end(); ++it){
		it->second.freq=0;
		it->second.label_marginal = vector <int> ();
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
void M3N::reload_training(char *train_file){
	int old_w_size = _w_size;
	generate_features(train_file);
	if(old_w_size!=_w_size)
		cerr << "Warning: w_size has changed from " << old_w_size << " -> " << _w_size << endl;
}
int M3N::get_index(vector<char *> &table, int start, int offset, string& comp_str, int comp_col, bool accept_equal, int ncol){
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

int M3N::make_feature_string(char *s, templet &pat, int i, int j, vector<char *> &table, int ncol)
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

bool M3N::add_x(vector<char *> &table){
	vector<int> y;
	int rows=table.size()/_cols;
	char s[1024];
	sequence seq;
	node* nod=_nodes.alloc(rows);
	seq.node_num=rows;
	seq.nodes=nod;

	_sequences.push_back(seq);
	for(int i=0;i<rows;i++){
		y.resize(y.size()+1);
		char *lab = get_pure_label(table[(i+1)*_cols-1], nod[i].bWeighted);
		vector_search(_tags, lab, y.back(), str_cmp());//get the tag of current node

		vector <int> keys(_gsize,0);
		{	// compute key at all orders
			int order;
			for(order=0; order<=_order; ++order){
				if(i-order<0)
					break;
				keys[order] = calc_key(&y[i-order],order+1);
				++_y_marginal[order][keys[order]];
			}
			nod[i].key=keys[order-1];
		}

		vector<clique*> clisp;//features that affect on current nodes
		int cur_group=0;
		vector<vector<int> > feature_vectors(_order+1);
		for(int j=0;j<_templets.size();j++){
			// if first y's offset < 0, skip current feature???
			templet &pat=_templets[j];
			if(pat.y[0]+i<0)
				continue;

			int k=make_feature_string(s, pat, i, j, table, _cols);

			if(s[0]){
				strcat(s,pat.words[k].c_str());

				//x obtained, insert x
				FeatureInfo *ppinfo;
				if(insert_x(s,&ppinfo,j)){
					int c=pow((double)_ysize,(int)pat.y.size());
					_w_size+=c;
				}

				//get clique
				feature_vectors[ppinfo->order].push_back(ppinfo->global_id);

				// increment feature occurrence
				++(ppinfo->freq);

				// add label marginal for the current feature
				++(ppinfo->label_marginal[keys[ppinfo->order]]);
			}
		}

		for(int order=0; order<=_order; ++order){	//create new cliques
			clique cli;
			vector<node*> ns;
			int key=0;
			for(int k=0;k<=order;k++){
				int node_id=i+k-order;
				ns.push_back(nod+node_id);
				key=key*_ysize+y[node_id];
			}
			node ** np=_clique_node.push_back(&ns[0],ns.size());
			cli.nodes=np;
			cli.node_num=ns.size();
			cli.key=key;

			vector<int> &feature_vector = feature_vectors[order];
			if(feature_vector.size())
				cli.fvector=_clique_feature.push_back(&feature_vector[0],feature_vector.size());
			else
				cli.fvector=NULL;
			cli.feature_num=feature_vector.size();
			cli.groupid=order;
			clique *new_clique=_cliques.push_back(&cli,1);
			clisp.push_back(new_clique);
		}

		//set node -> clique
		if(clisp.size())
			nod[i].cliques = _node_clique.push_back(&clisp[0],clisp.size());
		else
			nod[i].cliques=NULL;
		nod[i].clique_num =clisp.size();
	}
	return true;
}

bool M3N::insert_x(char *target, FeatureInfo **ppinfo, int template_id){
	map<char *, FeatureInfo, str_cmp>::iterator p;
	p=_xindex.find(target);
	if(p!=_xindex.end()){	// existing feature
		*ppinfo = &p->second;
		return false;
	}else{					// new feature
		char *q=_x_str.push_back(target);
		int order = _templets[template_id].y.size()-1;
		FeatureInfo info={
				_w_size,		// global_id
				0,				// freq
				template_id,
				order,			// order
				vector<int>(_templet_group[order].size(),0)				// label_marginal
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

void M3N::count_feature_at_each_order(map<int,int> &ret){
	ret.clear();
	for(map<char *, FeatureInfo, str_cmp>::iterator it=_xindex.begin(); it!=_xindex.end(); ++it)
		++ret[it->second.order];
}

void M3N::shrink_feature(){
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
	int prune_method = 0;
	if(!prune_params.empty())
		prune_method=prune_params[0];
    if(prune_method>=1){
    	// compute y distribution for all clique orders
		vector <vector <double> > y_marginal (_order+1);	// no entry can be 0
		for(int order=0; order<=_order; ++order)
			norm_P(y_marginal[order], _y_marginal[order], 1.0);

		// compute KL divergence for all features
		vector <feature_entry> entries;
		for(it=_xindex.begin(); it!= _xindex.end(); ++it){
			char *feature_name = it->first;
			FeatureInfo &info = it->second;
			vector <int> &can_be_zero = info.label_marginal;
			templet t = _templets[info.templet_id];
			//feature_entry e={feature_name, (!t.x.size()?1.0e10:KL_div(y_marginal[info.order],can_be_zero)*sum(can_be_zero))};	// default feature will NOT be prune away
			feature_entry e={feature_name, KL_div(y_marginal[info.order],can_be_zero)*sum(can_be_zero)};		// default feature might be prune away
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
				old2new[info.global_id] = pair<int,int>(new_lambda_size,info.order);
				info.global_id = new_lambda_size;
				info.label_marginal.clear();						// to save memory
				new_xindex[it->first] = info;
				new_lambda_size += _y_marginal[info.order].size();
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
				info.order=min(info.order,current_max_order);
				if(info.order==_order){		// we are at max order
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
				old2new[info.global_id] = pair<int,int>(new_lambda_size,info.order);
				info.global_id = new_lambda_size;
				info.label_marginal.clear();						// to save memory
				new_xindex[it->first] = info;
				new_lambda_size += _y_marginal[info.order].size();
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
				info.order=min(info.order,current_max_order);
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
				old2new[info.global_id] = pair<int,int>(new_lambda_size,info.order);
				info.global_id = new_lambda_size;
				info.label_marginal.clear();						// to save memory
				new_xindex[it->first] = info;
				new_lambda_size += _y_marginal[info.order].size();
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
			old2new[info.global_id] = pair<int, int>(new_lambda_size, info.order);
			info.global_id = new_lambda_size;
			new_xindex[it->first] = info;
			new_lambda_size += (int) pow(double(_ysize), info.order + 1);
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
				old2new[info.global_id] = pair<int,int>(new_lambda_size,info.order);
				info.global_id = new_lambda_size;
				new_lambda_size += (int)pow(double(_ysize),gram_num);
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
	int i,j,k,ii;
	for(i=0;i<_sequences.size();i++)			// for each sequence
	{
		sequence &seq=_sequences[i];
		for(j=0;j<seq.node_num;j++)				// for each node
		{
			node &nod=seq.nodes[j];
			vector <vector<int> > newfs(_order+1);
			for(k=0;k<nod.clique_num;k++)		// for each clique
			{
				if(!nod.cliques[k])
					continue;
				clique &cli=*nod.cliques[k];
				for(ii=0;ii<cli.feature_num;ii++)	// for each feature
				{
					iter = old2new.find(cli.fvector[ii]);
					if(iter != old2new.end())
						newfs[iter->second.second].push_back(iter->second.first);
				}
			}
			for(k=0; k<=_order; ++k)
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
	for(i=0;i<_sequences.size();i++)
	{
		sequence &seq=_sequences[i];
		for(j=0;j<seq.node_num;j++)
		{
			node &nod=seq.nodes[j];
			for(k=0;k<nod.clique_num;k++)
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
	for(int order=0; order<=_order; ++order)
		cerr << "clique " << order << " : " << orig_num_feat_each_clique[order]
			 << " -> " << num_feat_each_clique[order] << endl;
    return;
}

void M3N::initialize(){
	int i,j;
	//set key path for all nodes
	_node_anum=pow((double)_ysize,_order);
	_path_num=_node_anum*_ysize;
	if(_kernel_type==LINEAR_KERNEL){
		if(!_w){//if _w != NULL => _w has been initalized with model file
			_w=new double[_w_size];
			memset(_w,0,sizeof(double)*_w_size);
		}
	}
	//initialize mu
	if(!_mu){
		vector<double> mu;// marginal probabilities for all paths
		for(i=0;i<_sequences.size();i++){
			_sequences[i].mu_index=mu.size();
			for(j=0;j<_sequences[i].node_num;j++){
				int mu_size=mu.size();
				mu.resize(mu_size+_path_num,0);
				mu[mu_size+_sequences[i].nodes[j].key]=1;
			}
		}
		_mu_size=mu.size();
		_mu=new double[_mu_size];
		memcpy(_mu,&mu[0],sizeof(double)*_mu_size);
	}else{
		//relearn: assign seq.mu_index
		int mu_size=0;
		for(i=0;i<_sequences.size();i++){
			_sequences[i].mu_index=mu_size;
			mu_size+=_sequences[i].node_num*_path_num;
		}
	}
	_head_offset=-log((double)_ysize)*_order;
	_clique_kernel.resize(_gsize);
	_path_kernel.resize(_path_num*_path_num);
}

bool M3N::find_violation(sequence &seq, double &kkt_violation){
	int i,j;
	kkt_violation=0;

	forward_backward_viterbi(_ysize,_order,seq.node_num,_alpha_lattice,_optimum_alpha_lattice,_optimum_alpha_paths,_head_offset,LOGZERO,LOGZERO,INF);
	forward_backward_viterbi(_ysize,_order,seq.node_num,_v_lattice,_optimum_v_lattice,_optimum_v_paths,0,0,-INF,INF);
	
	vector<int> v_index;
	vector<double> vyc;
	vector<int> a_index;
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
		v_index.clear();
		vyc.clear();
		a_index.clear();
		ayc.clear();
		vyc.assign(_optimum_v_lattice.begin()+i*_path_num,_optimum_v_lattice.begin()+(i+1)*_path_num);
		merge_sort(vyc,v_index);
		ayc.assign(_optimum_alpha_lattice.begin()+i*_path_num,_optimum_alpha_lattice.begin()+(i+1)*_path_num);
		merge_sort(ayc,a_index);


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

			if(ayc[j]==LOGZERO && vyc[j]>v_yc+EPS){//violation=1;
				if(kkt_violation<vyc[j]-v_yc){
					kkt_violation=vyc[j]-v_yc;
					y1=i*_path_num+j;
					//assert(a_yc>LOGZERO);
					y2=i*_path_num+a_yc_index;
					_path1=_optimum_v_paths[y1];
					_path2=_optimum_alpha_paths[y2];
				}
			}else if(ayc[j]>LOGZERO && vyc[j]<v_yc-EPS){//violation=2;
				if(kkt_violation<v_yc-vyc[j]){
					kkt_violation=v_yc-vyc[j];
					y1=i*_path_num+j;
					//assert(v_yc>vyc[j]);
					y2=i*_path_num+v_yc_index;
					_path1=_optimum_alpha_paths[y1];
					_path2=_optimum_v_paths[y2];
				}
			}
		}
	}
	if(y1!=-1)
		return true;
	return false;
}
void M3N::build_alpha_lattice(sequence &seq){
	//_alpha_lattice[i*_path_num+j] is the log distribution on the marginal probability
	int i,j;
	memcpy(&_alpha_lattice[0],_mu+seq.mu_index,seq.node_num*_path_num*sizeof(double));
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
void M3N::build_v_lattice(sequence &seq){
	int i,j,k,ii,jj;
	//build _v_lattice
	if(_model==TEST_MODEL && _v_lattice.size()<seq.node_num*_path_num)
		_v_lattice.resize(seq.node_num*_path_num);
	fill(_v_lattice.begin(),_v_lattice.end(),0);
	if(_kernel_type==LINEAR_KERNEL){
		//LEARN_MODEL: v(x,y)=w'f(x,y)+l(x,y)=\sum_t w'f(x,y,t)+l(x,y,t)=\sum_t phi(x,y,t)
		//TEST_MODEL:  v(x,y)=w'f(x,y)       =\sum_t w'f(x,y,t)
		for(i=0;i<seq.node_num;i++){				// for each node
			//build phi(x,y,t) for each path
			node &n1=seq.nodes[i];
			for(j=0;j<n1.clique_num;j++){			// for each clique
				clique &cli=*(n1.cliques[j]);
				vector<vector<int> > &group=_templet_group[cli.groupid];
				for(k=0;k<cli.feature_num;k++){		// for each feature
					for(ii=0;ii<group.size();ii++){
						for(jj=0;jj<group[ii].size();jj++){
							_v_lattice[i*_path_num+group[ii][jj]]+=_w[cli.fvector[k]+ii];
						}
					}
				}
			}
			if(_model==LEARN_MODEL){
				//assert(n1.factor==1 || n1.factor==2);
				for(j=0;j<_path_num;j++){
					int ref = n1.key%_ysize;
					int sys = j%_ysize;
					if(ref!=sys)
						_v_lattice[i*_path_num+j]+=(n1.bWeighted?_loss[ref][sys]*_B:_loss[ref][sys]);
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
		for(i=0;i<seq.node_num;i++){
			//build phi(x,y,t) for each path
			node &n1=seq.nodes[i];
			//calculate n1's kernel
			for(j=0;j<_sequences.size();j++){
				if(_print_level>1)
					cerr<<"\t\tkernel compute: k(seq["<<i<<"],seq["<<j<<"])"<<endl;
				for(k=0;k<_sequences[j].node_num;k++){
					node &n2=_sequences[j].nodes[k];
					get_kernel(n1,n2);//calculate kernels
					//calculate v
					for(ii=0;ii<_path_num;ii++){
						_v_lattice[i*_path_num+ii]+=_path_kernel[ii*_path_num+n2.key];
						for(jj=0;jj<_path_num;jj++){
							_v_lattice[i*_path_num+ii]-=_path_kernel[ii*_path_num+jj]*_mu[_sequences[j].mu_index+k*_path_num+jj];
						}
					}
				}
			}
			for(j=0;j<_path_num;j++){
				_v_lattice[i*_path_num+j]*=_C;
				if(_model==LEARN_MODEL)
					if(j%_ysize!=n1.key%_ysize)
						_v_lattice[i*_path_num+j]++;
			}
		}
	}
}
void M3N::smo_optimize(sequence &seq){
	
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
		map <int,pair<int,int> > fvec_pair;
		for(int i=0;i<seq.node_num;i++){
			if(_path1[i]==_path2[i])
				continue;

			int J = seq.nodes[i].clique_num;
			clique **cs = seq.nodes[i].cliques;
			for(int j=0; j<J; ++j){
				int K=cs[j]->feature_num;
				int *fv=cs[j]->fvector;
				vector<int> &path2cliy = _path2cliy[cs[j]->groupid];
				for(int k=0; k<K; ++k){
					++fvec_pair[fv[k]+path2cliy[_path1[i]]].first;
					++fvec_pair[fv[k]+path2cliy[_path2[i]]].second;
				}
			}
		}

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
		_mu[i*_path_num+_path1[i]+seq.mu_index]+=delta;
		_mu[i*_path_num+_path2[i]+seq.mu_index]-=delta;
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
		for(int i=0;i<seq.node_num;i++){
			for(int j=0;j<seq.nodes[i].clique_num;j++){
				if(!seq.nodes[i].cliques[j])
					continue;
				clique &cli=*(seq.nodes[i].cliques[j]);
				for(int k=0;k<cli.feature_num;k++){
						_w[cli.fvector[k]+_path2cliy[cli.groupid][_path1[i]]]-=delta*_C;
						_w[cli.fvector[k]+_path2cliy[cli.groupid][_path2[i]]]+=delta*_C;
				}
			}
		}
	}
}


void M3N::sort_feature(sequence &seq){
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
void M3N::assign_tag(sequence &seq, vector<int> &node_tag)
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
void M3N::node_margin(sequence &seq, vector<vector<double> >&node_p,vector<double> &alpha, vector<double> &beta, double &z)
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
double M3N::path_cost(sequence &seq, vector<double>& lattice){
	int i;
	double c=0;
	for(i=0;i<seq.node_num;i++)
		c+=lattice[_path_num*i+seq.nodes[i].key];
	return c;
}
vector <double> M3N::compute_hypo_score(vector<char*> &table, vector<vector<char*> > &tags, int ncol){
	vector <double> costs;
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
	}
	return costs;
}
void M3N::tag(sequence &seq, vector<vector<int> > &best_tag_indices){	// from v-lattice
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
void M3N::tag(vector<char*> &table, vector<string> &best_tag, vector<vector<double> > &marginal, int ncol){	// compute posterior
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
	node_margin(seq, marginal, alpha, beta, z);
}
void M3N::tag(vector<char*> &table, vector<vector<string> > &best_tag,vector<vector<vector<double> > > &edgep, int ncol){
	edgep.clear();
	_model=TEST_MODEL;
	sequence seq;
	generate_sequence(table,seq,ncol);
	sort_feature(seq);
	build_v_lattice(seq);
	vector<vector<int> > best_path;
	int rows=table.size()/ncol;
	edgep.resize(rows);
	vector<vector<int> > valid_v(rows);
	
//	if(_nbest==1){
		viterbi(seq.node_num, _order, _ysize, _nbest, _v_lattice, best_path);
		int i,j;
		for(i=0;i<best_path.size();i++){
			vector<string> cur_tag(seq.node_num);
			for(j=0;j<seq.node_num;j++){
				cur_tag[j]=_tags[best_path[i][j]];
			}
			best_tag.push_back(cur_tag);
		}
/*	}else{
		for(int i=0;i<rows;i++){
			valid_v[i].resize(_ysize,0);
			edgep[i].resize(_ysize);
			for(int y=0;y<_ysize;y++){
				edgep[i][y].resize(_ysize,-INF);
				for(int z=0;z<_ysize;z++){
					edgep[i][y][z]=_v_lattice[i*_path_num+y*_ysize+z];
				}
			}
		}
		int i,j;
		for(int n=1;n<_nbest;n++){
			viterbi(seq.node_num, _order, _ysize, 2, _v_lattice, best_path);
			int y1,y2;
			int z1,z2;
			for(i=0;i<rows;i++){
				valid_v[i][best_path[0][i]]=1;
				if(i==0){
					y1=0;z1=0;
				}else{
					z1=best_path[1][i-1];
					y1=best_path[0][i-1];
				}
				y2=best_path[0][i];
				z2=best_path[1][i];
				if(y1!=z1 || y2!=z2){
					_v_lattice[i*_path_num+y1*_ysize+y2]=-INF;
				}
			}
			best_path.clear();
		}
		/*
		for(int i=0;i<rows;i++){
			if(i==0){
				for(int y=0;y<_ysize;y++){
					if(valid_v[0][y]==0)
						edgep[0][0][y]=-INF;
				}
				for(int y=1;y<_ysize;y++){
					for(int z=0;z<_ysize;z++){
						edgep[0][y][z]=-INF;
					}
				}
			}else{
				for(int y=0;y<_ysize;y++){
					for(int z=0;z<_ysize;z++){
						if(valid_v[i][z]==0 || valid_v[i-1][y]==0)
							edgep[i][y][z]=-INF;
					}
				}
			}
		}
	}*/
}
void M3N::generate_sequence(std::vector<char*> &table, sequence &seq, int ncol){
	int rows=table.size()/ncol;
	char s[1024];
	node* nod=_test_nodes.alloc(rows);
	seq.node_num=rows;
	seq.nodes=nod;
	for(int i=0;i<rows;i++)
	{
		nod[i].key=0;//random initialize
		vector<clique*> clisp;//features that affect on current nodes
		int cur_group=0;
		vector<vector<int> > feature_vectors(_order+1);
		for(int j=0;j<_templets.size();j++)
		{
			//get first y's offset
			templet &pat=_templets[j];
			if(pat.y[0]+i<0)
				continue;

			int k=make_feature_string(s, pat, i, j, table, ncol);

			if(s[0]){
				strcat(s,pat.words[k].c_str());
				//x obtained, insert x
				map<char *, FeatureInfo, str_cmp>::iterator it;
				it=_xindex.find(s);
				if(it!=_xindex.end())
					feature_vectors[it->second.order].push_back(it->second.global_id);
			}
		}
		for(int order=0; order<=_order; ++order){
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

bool M3N::load_model(const string& model_file){
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
	_templet_group.resize(_gsize);
	cerr<<"template number: "<<_templets.size()<<endl;

	//get ysize
	fin.getline(line,MAXSTRLEN-1);
	_ysize=atoi(line);
	_tags.resize(_ysize);
	for(i=0;i<_ysize;i++){
		fin.getline(line,MAXSTRLEN-1);
		char *q=_tag_str.push_back(line);
		_tags[i]=q;
	}
	cerr<<"tags number: "<<_ysize<<endl;
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
	for(i=0;i<x_num;i++){
		fin.getline(line,MAXSTRLEN-1);
		vector<char *> columns;
		split_string(line,"\t",columns);
		char *q=_x_str.push_back(columns[0]);
		FeatureInfo info;
		info.global_id = atoi(columns[1]);
		info.order = atoi(columns[2]);
		info.label_marginal = vector <int> (_ysize,0);
		if(columns.size()>=5){	// optionally load feature occurrence frequency and templet_id
			info.freq = atoi(columns[3]);
			info.templet_id = atoi(columns[4]);
		}
		_xindex[q]=info;
	}

	fin.getline(line,MAXSTRLEN-1);
	fin.getline(line,MAXSTRLEN-1);
	if(_kernel_type==LINEAR_KERNEL){
		//load obj
		_obj=atof(line);
		fin.getline(line,MAXSTRLEN-1);
		//load _w
		_w_size=atoi(line);
		_w=new double[_w_size];
		for(i=0;i<_w_size;i++){
			fin.getline(line,MAXSTRLEN-1);
			_w[i]=atof(line);
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
		_get_kernel=&M3N::linear_kernel;
		_get_kernel_list=&M3N::linear_kernel;
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
			_get_kernel=&M3N::poly_kernel;
			_get_kernel_list=&M3N::poly_kernel;
		}else if(_kernel_type==NEURAL_KERNEL){
			_get_kernel=&M3N::neural_kernel;
			_get_kernel_list=&M3N::neural_kernel;
		}else if(_kernel_type==RBF_KERNEL){
			_get_kernel=&M3N::rbf_kernel;
			_get_kernel_list=&M3N::rbf_kernel;
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

void M3N::print(ostream &fout){
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
		double val = std_deviation(&_w[feature_id], n_weights[feature_id]);
		g_map.insert(make_pair(val,feature_str));
	}

	// write
	cerr << "writing 1... _tags.size()="<< _tags.size() << endl;
	fout << "tags: ";
	for(int x=0; x<_tags.size(); x++)
		fout << _tags[x] << " ";
	fout << endl;

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
		fout << " order=" <<  info.order;
		if(info.freq){
			fout << " freq=" << info.freq << " tid=" << info.templet_id;
		}
		if(sum(info.label_marginal)){
			fout << " label-marginal=";
			for(int x=0; x<_ysize; x++){
				fout << ' ' << info.label_marginal[x];
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

