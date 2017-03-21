#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>

using namespace std;

int main(int argc, char *argv[]){
	string line;
	vector<string> lines;
	ifstream infile(argv[1]);
	ofstream outfile(argv[2]);
	vector<string>::size_type i;

	lines.clear();
	if(infile.is_open()){
		while(getline(infile, line)){
			lines.push_back(line);
		}
	}

	sort(lines.begin(), lines.end());
	for(i=0;i!=lines.size();i++)
		outfile << lines[i] << endl;

	infile.close();
	outfile.close();
	return 0;
}
