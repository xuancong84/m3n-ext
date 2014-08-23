#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <cstring>
#include <cstdlib>
#include <assert.h>
#include "gzstream.h"
#include "m3n.h"
#include "fun.h"


using namespace std;

void vvs2vvc(vector<vector<char*> > &vvc, vector<vector<string> > &vvs){
	vvc.clear();
	for(int x=0; x<vvs.size(); x++){
		vvc.push_back(vector<char*>());
		for(int y=0; y<vvs[x].size(); y++)
			vvc.back().push_back((char*)vvs[x][y].c_str());
	}
}

void test(FM3N *m, istream &fin, ostream &fout, int margin, int nbest, int nbest_score){
	vector < char* >table;
	int total=0;
	vector <int> error(m->_y_max_id+1,0);
	int index=0;

	char line[10000];
	int ncol=0,nlines=0;
	vector <string> buf;
	if(margin){
		for(int y=0; y<=m->_y_max_id; ++y){
			fout << "label" << y;
			for(int x=0; x<m->_tags[y].size(); ++x)
				fout << ' ' << m->_tags[y][x];
		}
		fout << endl;
		fout.flush();
	}
	while(fin.getline(line,9999)){
		if(line[0]){
			buf.push_back(line);
			if(!split_string((char*)buf.back().c_str(),"\t",table))
				goto L0;
			if(ncol==0)
				ncol=table.size();
			if((++nlines)%10000==0)
				cerr << ".";
		}else if(table.size()){
L0:
			index++;
			if(margin){		// output marginal probabilities
				vector < vector <double> > marginals;
				vector <string> ys;
				bool dummy;
				m->tag(table,ys,marginals,ncol);
				for(int x=0,X=marginals.size(); x<X; ++x){
					vector <double> &marginal = marginals[x];
					fout << ys[x];
					for(int y=0, Y=marginal.size(); y<Y; ++y)
						fout << ' ' << marginal[y];
					fout << endl;

					// compute score
					++total;
					if(string(get_pure_label(table[(x+1)*ncol-1],dummy))!=ys[x])
						error[0]++;
				}
			}else{
				vector < vector < vector < string > > > y;
				m->tag(table,y,ncol);
				int table_cols=ncol;
				int table_rows=table.size()/ncol;
				total+=table_rows;
				if(nbest==1 && nbest_score==0){
					for(int i=0;i < table_rows ; i++)
					{
						int j;
						fout<<table[i*table_cols];
						for(j=1; j<table_cols; j++)
							fout<<'\t'<<table[i*table_cols+j];
						for(j=m->_y_max_id; j>=0; --j){
							fout<<'\t'<<y[0][i][j];
							if(table[(i+1)*ncol-1-j]!=y[0][i][j])
								error[j]++;
						}
						fout<<endl;
					}
				}else{	// n-best output
					/*TODO
					vector <double> scores;
					if(nbest_score==1){
						vector <vector <char*> > lab_seqs;
						vvs2vvc(lab_seqs,y);
						scores = m->compute_hypo_score(table,lab_seqs,ncol);
					}
					for(int j=0; j<abs(nbest) && j<y.size(); j++){
						if(!scores.empty())
							fout << scores[j] << "\t";
						for(int i=0; i<table_rows; i++){
							fout << (i==0?"":"\t") << y[j][i];
							if(!j && string(table[(i+1)*ncol-1])!=y[j][i])
								error++;
						}
						fout << endl;
					}*/
				}
			}
			table.clear();
			buf.clear();
			fout<<endl;
			fout.flush();
		}else{
			table.clear();
			buf.clear();
			fout<<endl;
			fout.flush();
		}
	}
	if(table.size()){
		goto L0;
	}
	cerr << "Done!" << endl;
	for(int x=0; x<=m->_y_max_id; ++x)
		cerr << "label precision at layer "<<x<<" : "<<(1-(double)error[x]/total)<<endl;
}

void score(FM3N *m, istream &fin, istream &fnbest, ostream &fout, int mode){
	vector <char*> table;
	vector <vector <char*> > lab_seqs;
	int total=0;
	int error=0;
	int index=0;

	char line[10000];
	int ncol=0,nlines=0;
	vector <string> buf, buf2;
	string hyp;

	while(getline(fnbest,hyp)){
		if(hyp.empty()){
			// load 1 linegroup from input
			table.clear();
			string line;
			buf.clear();
			while(getline(fin,line)){
				buf.push_back(line);
				if(!split_string((char*)buf.back().c_str(),"\t",table))
					break;
				if(ncol==0)
					ncol=table.size();
				if((++nlines)%10000==0)
					cerr << ".", cerr.flush();
			}
			vector <double> scores = m->compute_hypo_score(table,lab_seqs,ncol);
			for(int x=0; x<scores.size(); x++)
				fout << scores[x] << endl;
			fout << endl;
			fout.flush();
			table.clear();
			lab_seqs.clear();
		}else{
			buf2.push_back(hyp);
			lab_seqs.push_back(vector<char*>());
			split_string((char*)buf2.back().c_str(),"\t",lab_seqs.back());
		}
	}

	cerr << "Done!" << endl;
}

//build learn
int main_learn(int argc, char *argv[]){
const char* learn_help="\
format:\n\
m3n_learn template_file_name(.gz) train_file_name(.gz) output_model_file_name(.gz)\n\
option type   default   meaning\n\
-c     double 1         Slack variables penalty factor\n\
-b     double 2         Node biasing factor\n\
-f     int    0         Frequency threshold.\n\
-k     int    0         Kernel type. \n\
                        0: linear kernel <a,b>\n\
                        1: polynomial kernel (s*<a,b>+r)^d\n\
                        2: rbf kernel exp{-s*||a-b||^2}\n\
                        3: neural kernel tanh(s*<a,b>+r)\n\
-s     double 1         \n\
-d     int    1         \n\
-r     double 0         \n\
-i     int    10        Max iteration number. \n\
-e     double 0.000001  Controls training precision \n\
-o     int    0         With higher value, more information is printed.\n\
-l     string loss.txt  loss weight file.\n\
-x                      relearn, overwriting input model.\n\
-a     int    0         label is at 0:last/1:first column (for reading only)\n\
-p     int[]  0         pruning method: 0=freq  1,max_param:KL_div  2,max_param,#-max-order-features:\n\
-m     string ''        output model path per iteration\n\
-pf    string ''        print feature to file\n";
//initial learning parameters
	string train_file;
	string templet_file;
	string model_file;
	string iter_model_file;
	string print_feature_file;
	string input_model;
	string loss_file="";
	bool relearn=false;			//-x
	char C[100]="1";			//-c
	char B[100]="2";			//-b
	char freq_thresh[100]="0";	//-f
	char kernel_type[100]="0";	//-k
	char kernel_s[100]="1";		//-s
	char kernel_d[100]="1";		//-d
	char kernel_r[100]="0";		//-r
	char max_iter[100]="10";	//-i
	char eta[100]="0.000001";	//-e
	char print_level[100]="0";	//-o
	//get learning parameters, and check
	int i=2;
	int read_mode=0;
	vector <int> prune_params;
	int step=0;//0: next load templet_file, 1: next load train_file, 2: next_load model_file
	while(i<argc){
		if(!strcmp(argv[i],"-p")){
			if(i+1==argc){
				cerr<<"-p parameter empty"<<endl;
				return 1;
			}
			char *s = argv[i+1];
			vector <char*> params;
			split_string(s,",",params);
			for(int x=0; x<params.size(); ++x)
				prune_params.push_back(atoi(params[x]));
			i+=2;
		}else if(!strcmp(argv[i],"-l")){
			if(i+1==argc){
				cerr<<"-l parameter empty"<<endl;
				return 1;
			}
			loss_file=argv[i+1];
			i+=2;
		}else if(!strcmp(argv[i],"-m")){
			if(i+1==argc){
				cerr<<"-m parameter empty"<<endl;
				return 1;
			}
			iter_model_file=argv[i+1];
			i+=2;
		}else if(!strcmp(argv[i],"-pf")){
			if(i+1==argc){
				cerr<<"-pf parameter empty"<<endl;
				return 1;
			}
			print_feature_file=argv[i+1];
			i+=2;
		}else if(!strcmp(argv[i],"-x")){
			relearn=true;
			i++;
		}else if(!strcmp(argv[i],"-h")){
			cerr<<learn_help<<endl;
			return 0;
		}else if(!strcmp(argv[i],"-c")){
			if(i+1==argc){
				cerr<<"-c parameter empty"<<endl;
				return 1;
			}
			strcpy(C,argv[i+1]);
			if(atof(C)<=0){
				cerr<<"invalid -c parameter"<<endl;
				return 1;
			}
			i+=2;
		}else if(!strcmp(argv[i],"-b")){
			if(i+1==argc){
				cerr<<"-b parameter empty"<<endl;
				return 1;
			}
			strcpy(B,argv[i+1]);
			i+=2;
		}else if(!strcmp(argv[i],"-f")){
			if(i+1==argc){
				cerr<<"-f parameter empty"<<endl;
				return 1;
			}
			strcpy(freq_thresh,argv[i+1]);
			if(atoi(freq_thresh)<0){
				cerr<<"invalid -f parameter"<<endl;
				return 1;
			}
			i+=2;
		}else if(!strcmp(argv[i],"-k")){
			if(i+1==argc){
				cerr<<"-k parameter empty"<<endl;
				return 1;
			}
			strcpy(kernel_type,argv[i+1]);
			if(atoi(kernel_type)<0||atoi(kernel_type)>3){
				cerr<<"invalid -k parameter"<<endl;
				return 1;
			}
			i+=2;
		}else if(!strcmp(argv[i],"-s")){
			if(i+1==argc){
				cerr<<"-s parameter empty"<<endl;
				return 1;
			}
			strcpy(kernel_s,argv[i+1]);
			if(atof(kernel_s)<0){
				cerr<<"invalid -s parameter"<<endl;
				return 1;
			}
			i+=2;
		}else if(!strcmp(argv[i],"-d")){
			if(i+1==argc){
				cerr<<"-d parameter empty"<<endl;
				return 1;
			}
			strcpy(kernel_d,argv[i+1]);
			if(atoi(kernel_d)<1){
				cerr<<"invalid -d parameter"<<endl;
				return 1;
			}
			i+=2;
		}else if(!strcmp(argv[i],"-a")){
			if(i+1==argc){
				cerr<<"-a parameter empty"<<endl;
				return 1;
			}
			read_mode = atoi(argv[i+1]);
			i+=2;
		}else if(!strcmp(argv[i],"-r")){
			if(i+1==argc){
				cerr<<"-r parameter empty"<<endl;
				return 1;
			}
			strcpy(kernel_r,argv[i+1]);
			if(atof(kernel_r)<0){
				cerr<<"invalid -r parameter"<<endl;
				return 1;
			}
			i+=2;
		}else if(!strcmp(argv[i],"-i")){
			if(i+1==argc){
				cerr<<"-i parameter empty"<<endl;
				return 1;
			}
			strcpy(max_iter,argv[i+1]);
			if(atoi(max_iter)<0){
				cerr<<"invalid -i parameter"<<endl;
				return 1;
			}
			i+=2;
		}else if(!strcmp(argv[i],"-e")){
			if(i+1==argc){
				cerr<<"-e parameter empty"<<endl;
				return 1;
			}
			strcpy(eta,argv[i+1]);
			if(atof(eta)<0){
				cerr<<"invalid -e parameter"<<endl;
				return 1;
			}
			i+=2;
		}else if(!strcmp(argv[i],"-o")){
			if(i+1==argc){
				cerr<<"-o parameter empty"<<endl;
				return 1;
			}
			strcpy(print_level,argv[i+1]);
			if(atoi(print_level)<0){
				cerr<<"invalid -o parameter"<<endl;
				return 1;
			}
			i+=2;
		}else if(argv[i][0]=='-'){
			cerr<<argv[i]<<": invalid parameter"<<endl;
			return 1;
		}else if(step==0){
			templet_file=argv[i];
			i++;
			step++;
		}else if(step==1){
			train_file=argv[i];
			i++;
			step++;
		}else if(step==2){
			model_file=argv[i];
			i++;
			step++;
		}
	}
	//check necessary parameters
	if(templet_file.empty()){
		cerr<<"no template file"<<endl;
		return 1;
	}
	if(train_file.empty()){
		cerr<<"no train file"<<endl;
		return 1;
	}
	if(model_file.empty()){
		cerr<<"no model file"<<endl;
		return 1;
	}
	FM3N *m=new FM3N();
	m->read_mode = read_mode;
	m->prune_params = prune_params;
	m->set_para("C",C);
	m->set_para("B",B);
	m->set_para("freq_thresh",freq_thresh);
	m->set_para("kernel_type",kernel_type);
	m->set_para("kernel_s",kernel_s);
	m->set_para("kernel_d",kernel_d);
	m->set_para("kernel_r",kernel_r);
	m->set_para("max_iter",max_iter);
	m->set_para("eta",eta);
	m->set_para("print_level",print_level);
	m->iter_model_file = iter_model_file;
	if(print_feature_file!="")
		m->print_feature_file = OpenWrite(print_feature_file);
	m->learn(templet_file.c_str(), train_file.c_str(), model_file.c_str(), loss_file.c_str(), relearn);
	delete m;
	return 0;
}

int main_test(int argc, char *argv[]){
const char* test_help="\
format:\n\
m3n_test model_file_name(.gz) input_file_name(.gz) output_file_name(.gz)\n\
(Note: set filename to - for stdin/stdout)\n\
option type   default   meaning\n\
-m     int    0         output marginal probability, 0:none, 1:probability, 2:normalized-logP, 3:unnormalized-logP\n\
-n     int    1         output n best results (-ve value force output in n-best format)\n\
-s     int    0         output score for n best results, 0:no score, 1:v_lattice_score\n\
-a     int    0         label is at 0:last/1:first column (for reading only)\n\
";
//initial testing parameters
	string model_file;
	string key_file;
	string result_file;
	char margin[100]="0";		//-m
	char nbest[100]="1";		//-n
	int nbest_score=0;

	int read_mode=0;
	int i=2;
	int step=0;//0: next load model_file, 1: next load key_file,2: result_file
	while(i<argc){
		if(!strcmp(argv[i],"-m")){
			if(i+1==argc){
				cerr<<"-m parameter empty"<<endl;
				return 1;
			}
			strcpy(margin,argv[i+1]);
			i+=2;
		}else if(!strcmp(argv[i],"-n")){
			if(i+1==argc){
				cerr<<"-n parameter empty"<<endl;
				return 1;
			}
			strcpy(nbest,argv[i+1]);
			i+=2;
		}else if(!strcmp(argv[i],"-s")){
			if(i+1==argc){
				cerr<<"-s parameter empty"<<endl;
				return 1;
			}
			nbest_score=atoi(argv[i+1]);
			if(nbest_score<0 || nbest_score>1){
				cerr<<"invalid -s parameter"<<endl;
				return 1;
			}
			i+=2;
		}else if(!strcmp(argv[i],"-a")){
			if(i+1==argc){
				cerr<<"-a parameter empty"<<endl;
				return 1;
			}
			read_mode = atoi(argv[i+1]);
			i+=2;
		}else if(!strcmp(argv[i],"-h")){
			cerr<<test_help<<endl;
			return 0;
		}else if(step==0){
			model_file=argv[i];
			i++;
			step++;
		}else if(step==1){
			key_file=argv[i];
			i++;
			step++;
		}else if(step==2){
			result_file=argv[i];
			i++;
			step++;
		}else{
			cerr<<argv[i]<<": invalid parameter"<<endl;
			return 1;
		}
	}
	//check necessary parameters
	if(model_file.empty()){
		cerr<<"no model file"<<endl;
		return 1;
	}

	if(key_file.empty()){
		cerr<<"no input file"<<endl;
		return 1;
	}
	istream *fin=&cin;
	if(key_file!="-")
		fin=OpenRead(key_file.c_str());

	if(result_file.empty()){
		cerr<<"no output file"<<endl;
		return 1;
	}
	ostream *fout=&cout;
	if(result_file!="-")
		fout=OpenWrite(result_file.c_str());

	FM3N *m=new FM3N();
	m->read_mode=read_mode;
	m->set_para("margin",margin);
	m->set_para("nbest",nbest);
	m->load_model(model_file.c_str());
	test(m,*fin,*fout,atoi(margin),atoi(nbest),nbest_score);
	CloseIO(fout);
	delete m;
	return 0;
}

int main_score(int argc, char *argv[]){
const char* test_help="\
format:\n\
m3n_test model_file_name(.gz) input_file_name(.gz) input_nbest_file(.gz) output_score_file(.gz)\n\
(Note: set filename to - for stdin/stdout)\n\
option type   default   meaning\n\
-m     int    0         output marginal probability, 0:none, 1:probability, 2:normalized-logP, 3:unnormalized-logP\n\
-n     int    1         Output n best results.\n\
-a     int    0         label is at 0:last/1:first column (for reading only)\n\
";
//initial testing parameters
	string model_file;
	string nbest_file;
	string input_file;
	string result_file;
	char margin[100]="0";		//-m

	int read_mode=0;
	int i=2;
	int step=0;		//0: next load model_file; 1: next load input_file; 2: load nbest_file; 3: result_file
	while(i<argc){
		if(!strcmp(argv[i],"-m")){
			if(i+1==argc){
				cerr<<"-m parameter empty"<<endl;
				return 1;
			}
			strcpy(margin,argv[i+1]);
			i+=2;
		}else if(!strcmp(argv[i],"-a")){
			if(i+1==argc){
				cerr<<"-a parameter empty"<<endl;
				return 1;
			}
			read_mode = atoi(argv[i+1]);
			i+=2;
		}else if(!strcmp(argv[i],"-h")){
			cerr<<test_help<<endl;
			return 0;
		}else if(step==0){
			model_file=argv[i];
			i++;
			step++;
		}else if(step==1){
			input_file=argv[i];
			i++;
			step++;
		}else if(step==2){
			nbest_file=argv[i];
			i++;
			step++;
		}else if(step==3){
			result_file=argv[i];
			i++;
			step++;
		}else{
			cerr<<argv[i]<<": invalid parameter"<<endl;
			return 1;
		}
	}

	//check necessary parameters
	if(model_file.empty()){
		cerr<<"no model file"<<endl;
		return 1;
	}
	if(input_file.empty()){
		cerr<<"no input file"<<endl;
		return 1;
	}
	istream *fin = (input_file=="-" ? &cin : OpenRead(input_file.c_str()));

	if(nbest_file.empty()){
		cerr<<"no nbest file"<<endl;
		return 1;
	}
	istream *fnbest = (nbest_file=="-" ? &cin : OpenRead(nbest_file.c_str()));

	if(result_file.empty()){
		cerr<<"no output file"<<endl;
		return 1;
	}
	ostream *fout = (result_file=="-" ? &cout : OpenWrite(result_file.c_str()));

	FM3N *m=new FM3N();
	m->read_mode=read_mode;
	m->set_para("margin",margin);
	m->load_model(model_file.c_str());
	score(m,*fin,*fnbest,*fout,atoi(margin));
	CloseIO(fout);
	delete m;
	return 0;
}

int main_print(int argc, char *argv[]){
	const char* test_help="\
format:\n\
m3n_print model_file_name(.gz) output_file_name(.gz)\n\
option type   default   meaning\n\
-t     string NULL      Reload training file\n\
-m     int    0         Whether output marginal probability.\n\
-n     int    1         Output n best results.\n\
-a     int    0         label is at 0:last/1:first column (for reading only)\n\
";
	//initial testing parameters
	string model_file;
	string output_file;
	char margin[100]="0";		//-m
	char nbest[100]="1";		//-n
	char *reload_train = NULL;

	int read_mode=0;
	int i=2;
	int step=0;		//0: next load model_file, 1: next load key_file,2: result_file
	while(i<argc){
		if(!strcmp(argv[i],"-t")){
			if(i+1==argc){
				cerr<<"-t parameter empty"<<endl;
				return 1;
			}
			reload_train=argv[i+1];
			i+=2;
		}else if(!strcmp(argv[i],"-m")){
			if(i+1==argc){
				cerr<<"-m parameter empty"<<endl;
				return 1;
			}
			strcpy(margin,argv[i+1]);
			if(atoi(margin)!=0 && atoi(margin)!=1){
				cerr<<"invalid -m parameter"<<endl;
				return 1;
			}
			i+=2;
		}else if(!strcmp(argv[i],"-n")){
			if(i+1==argc){
				cerr<<"-n parameter empty"<<endl;
				return 1;
			}
			strcpy(nbest,argv[i+1]);
			if(atoi(nbest)<1){
				cerr<<"invalid -n parameter"<<endl;
				return 1;
			}
			i+=2;
		}else if(!strcmp(argv[i],"-a")){
			if(i+1==argc){
				cerr<<"-a parameter empty"<<endl;
				return 1;
			}
			read_mode = atoi(argv[i+1]);
			i+=2;
		}else if(!strcmp(argv[i],"-h")){
			cerr<<test_help<<endl;
			return 0;
		}else if(argv[i][0]=='-'){
			cerr<<argv[i]<<": invalid parameter"<<endl;
			return 1;
		}else if(step==0){
			model_file = argv[i];
			i++;
			step++;
		}else if(step==1){
			output_file = argv[i];
			i++;
			step++;
		}
	}

	//check necessary parameters
	if(model_file.empty()){
		cerr<<"no model file"<<endl;
		return 1;
	}
	if(output_file.empty())
		output_file="-";

	ostream &fout = *OpenWrite(output_file);

	FM3N *m=new FM3N();
	m->read_mode=read_mode;
	m->set_para("margin",margin);
	m->set_para("nbest",nbest);
	m->load_model(model_file.c_str());
	if(reload_train)
		m->reload_training(reload_train);
	m->print(fout);
	CloseIO(fout);
	delete m;
	return 0;
}

/*
int main_decode(int argc, char *argv[]){
	const char* test_help="\
format:\n\
m3n_test model_file_name <input >output\n\
option type   default   meaning\n\
-m     int    0         Whether output marginal probability.\n\
-n     int    1         Output n best results.\n\
";
	//initial testing parameters
	string model_file;
	char margin[100]="0";		//-m
	char nbest[100]="1";		//-n

	int i=2;
	int step=0;//0: next load model_file
	while(i<argc){
		if(!strcmp(argv[i],"-m")){
			if(i+1==argc){
				cerr<<"-m parameter empty"<<endl;
				return 1;
			}
			strcpy(margin,argv[i+1]);
			if(atoi(margin)!=0 && atoi(margin)!=1){
				cerr<<"invalid -m parameter"<<endl;
				return 1;
			}
			i+=2;
		}else if(!strcmp(argv[i],"-n")){
			if(i+1==argc){
				cerr<<"-n parameter empty"<<endl;
				return 1;
			}
			strcpy(nbest,argv[i+1]);
			if(atoi(nbest)<1){
				cerr<<"invalid -n parameter"<<endl;
				return 1;
			}
			i+=2;
		}else if(!strcmp(argv[i],"-h")){
			cerr<<test_help<<endl;
			return 0;
		}else if(argv[i][0]=='-'){
			cerr<<argv[i]<<": invalid parameter"<<endl;
			return 1;
		}else if(step==0){
			model_file = argv[i];
			i++;
			step++;
		}
	}
	//check necessary parameters
	if(!model_file[0]){
		cerr<<"no model file"<<endl;
		return 1;
	}

	M3N *m=new M3N();
	m->set_para("margin",margin);
	m->set_para("nbest",nbest);
	m->load_model(model_file.c_str());
	test(m,cin,cerr,atoi(margin),atoi(nbest));
	delete m;
	return 0;
}*/

int main(int argc, char *argv[]){
	cerr << "Version 0.11.8" << endl;
	if(argc>=2 && !strcmp(argv[1],"learn"))
		return main_learn(argc,argv);
	else if(argc>=2 && !strcmp(argv[1],"test"))
		return main_test(argc,argv);
	else if(argc>=2 && !strcmp(argv[1],"print"))
		return main_print(argc,argv);
	else if(argc>=2 && !strcmp(argv[1],"score"))
		return main_score(argc,argv);
	else
		cerr<<"invalid parameter, type 'm3n learn/test/print -h' for help"<<endl;
	return 0;
}
