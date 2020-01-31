#include <string>
#include <iostream>
#include <sstream>
#include <exception>
#include <vector>
#include <queue>
#include <map>
using namespace std;

class EdxException: public exception {

public:

    EdxException(string str): str(str) {}

    virtual const char* what() const noexcept {
        return str.c_str();
    }

private:
    string str{};
};

class Solution {
public:
    vector<int> findClosestElements(vector<int>& arr, int k, int x);
    int findPeakElement(vector<int>& nums);
};

vector<int> Solution::findClosestElements(vector<int>& arr, int k, int x){

    if (!arr.size() || !k){
        vector<int> result{};
        return result;
    }

    int left = 0;
    int right = arr.size() - 1;
    int mid{};
    while( (left + 1) < right ) {
        mid = (left + right) / 2;
        if (arr[mid] == x)
            break;
        else if(arr[mid] < x)
            left = mid;
        else if(arr[mid] > x)
            right = mid;
    }

    int leftIdx = mid;
    int rightIdx = mid;

    if (arr[mid] != x) {
        if ( ( x - arr[left] ) < ( arr[right] - x ) ) {
            leftIdx = left;
            rightIdx = left;
        }
        else {
            leftIdx = right;
            rightIdx = right;
        }
    }

    while((rightIdx+1)-leftIdx < k)
    {
        if( (leftIdx > 0))
        {
            if((rightIdx < arr.size()-1))
            {
                int rightEval = arr[rightIdx+1] -  x;
                int leftEval =  x - arr[leftIdx-1];
                if (leftEval > rightEval)
                    rightIdx += 1;
                else
                    leftIdx -= 1;
            }
            else {
                leftIdx -= 1;
            }
        }
        else if((rightIdx < arr.size()-1))
        {
            if( (leftIdx > 0))
            {
                int rightEval = arr[rightIdx+1] -  x;
                int leftEval =  x - arr[leftIdx-1];
                if (leftEval > rightEval)
                    rightIdx += 1;
                else
                    leftIdx -= 1;
            }
            else {
                rightIdx += 1;
            }
        }
        else {
            break;
        }
    }
    vector<int> result(arr.begin() + leftIdx, arr.begin() + rightIdx+1);
    return result;
}

int findPeakElement(vector<int>& nums){

    if(nums.size() < 2)
    {
        return (!nums.size()) ? -1 : 0;
    }

    //Let's check the bounders (upper and lower)...
    int left = 0;
    int right = nums.size() -1;
    if(nums[left] > nums[left+1])
        return left;
    if(nums[right] > nums[right-1])
        return right;

    //Let's check the rest 
    int mid{};
    while(left < right)
    {
        mid = (right + left) / 2;
        if(nums[mid] > nums[mid + 1])
        {
            right = mid;
        }
        else
        {
            left = mid + 1;
        }
    }
    return left;
}

struct SnailNode {
    SnailNode();
    SnailNode(int row, int col): row(row), col(col) {};
    int row{-1};
    int col{-1};
    pair<int, int> operator-(const SnailNode &other){
        return make_pair(row - row, col - col);
    }
    bool operator!=(const SnailNode &other){
        return (row != row || col != col);
    }
};

std::vector<int> snail(std::vector<std::vector<int>> snail_map) {

    vector<int> snailValPath{};
    if(!snail_map.size() || !snail_map.at(0).size())
        return snailValPath;

    vector<pair<int, int> > snail_pattern {
        pair<int, int>(0, 1),
        pair<int, int>(1, 0),
        pair<int, int>(0, -1),
        pair<int, int>(-1, 0)
    };

    size_t N = snail_map.size();

    vector<vector<bool> > visited(N);
    for(int i=0; i<N; i++)
        for(int j=0; j<N; j++)
            visited[i].push_back(false);


    auto genChild = [&snail_pattern, &N](SnailNode &node, pair<int,int> &prefPat){
        vector<SnailNode> childs;

        SnailNode childNode(node.row, node.col);
        int rowHop = node.row + prefPat.first;
        int colHop = node.col + prefPat.second;
        childNode.row = (rowHop < N && rowHop >= 0) ? rowHop : node.row;
        childNode.col = (colHop < N && colHop >= 0) ? colHop : node.col;
        if (childNode != node)
            childs.push_back(childNode);

        for(auto pat: snail_pattern){
            if (pat == prefPat)
                continue;
            SnailNode childNode(node.row, node.col);
            rowHop = node.row + pat.first;
            colHop = node.col + pat.second;
            childNode.row = (rowHop < N && rowHop >= 0) ? rowHop : node.row;
            childNode.col = (colHop < N && colHop >= 0) ? colHop : node.col;
            if (childNode != node)
                childs.push_back(childNode);
        }
        return childs;
    };

    vector<SnailNode> startPath{SnailNode(0,0)};
    queue<vector<SnailNode> > pathQueue;
    pathQueue.push(startPath);

    while(pathQueue.size())
    {
        vector<SnailNode> snailPath = pathQueue.front();
        pathQueue.pop();

        SnailNode node = *(snailPath.end() - 1);
        pair <int, int> prefPat = (snailPath.size() < 2) ? 
                                        snail_pattern.at(0) :
                                        node - *(snailPath.end() - 2);

        snailValPath.push_back(snail_map[node.row][node.col]);

        visited[node.row][node.col] = true;
        vector<SnailNode> childs = genChild(node, prefPat);

        for(auto child: childs) {
            if(!visited[child.row][child.col]){
                vector<SnailNode> newPath = snailPath;
                newPath.push_back(child);
                pathQueue.push(newPath);
                break;
            }
        }
    }

    for (auto el: snailValPath)
        cout << el << ", ";
    cout << endl;
    return snailValPath;
}

class DateTime {

public:
    DateTime() {}
    virtual ~DateTime() {}


    void formatDateTime(int seconds)
    {
        for(auto i=0; i<dtStruct.size(); i++)
            formatHelper(seconds, dtStruct[i].second, dtStruct[i].first);
    }

    friend ostream& operator<<(ostream &out, DateTime &other) {
        string format = other.getOneStringFormat();
        return (cout << format);
    }

    string getOneStringFormat(){

        vector<string> format;
        string s_form{};
        string sAnd{" and "};
        string comma{", "};

        for(auto el: dtStruct)
            if(el.second.first)
                format.push_back(el.second.second);

        if (format.size() == 1)
            return format.at(0);

        for(auto el=format.begin(); el != format.end() - 1; el++ )
        {
            s_form = (el==format.begin()) ?
                        (s_form + *el) :
                        (s_form + comma + *el);
        }

        return  s_form + sAnd + *(format.end() - 1);
    }

private:
    template<typename T>
    void formatHelper(int &t, T &t2, long long limit=1) {
        if (t >= limit) {
            string plural{'s'};
            t2.first = t / limit;
            t2.second = (t2.first > 1) ?
                            to_string(t2.first) + t2.second + plural:
                            to_string(t2.first) + t2.second;
            t %= limit;
        }
    }

private:
    vector<pair<int, pair<int, string> > > dtStruct{
        {365*24*3600, {0, " year"}},
        {24*3600, {0, " day"}},
        {3600, {0, " hour"}},
        {60,  {0, " minute"}},
        {1, {0, " second"}},
    };
};

string format_duration(int seconds)
{
    string dtFormat{"now"};
    if (!seconds)
        return dtFormat;

    DateTime dt;
    dt.formatDateTime(seconds);

    return dt.getOneStringFormat();
}

#include <cctype>
#include <stack>

string sum_strings(string a, string b) {
    stack<int> aStack;
    stack<int> bStack;

    for(auto ch: a) {
        if(!(ch - '0') && !aStack.size())
            continue;
        aStack.push((ch - '0'));
    }

    for(auto ch: b){
        if(!(ch - '0') && !bStack.size())
            continue;
        bStack.push((ch - '0'));
    }

    stack<int> sumStack;
    int aTop{}, bTop{}, retain{}, cycSum{}, cycMod{};
    while(bStack.size() || aStack.size() || retain)
    {
        if(aStack.size()){
            aTop = aStack.top();
            aStack.pop();
        }
        if(bStack.size()){
            bTop = bStack.top();
            bStack.pop();
        }
        cycSum = aTop + bTop + retain;
        if(cycSum >= 10)
        {
            retain = cycSum / 10;
            sumStack.push(cycSum % 10);
        }
        else {
            sumStack.push(cycSum);
            retain = 0;
        }
        aTop = 0;
        bTop = 0;
    }

    stringstream stream;
    while(sumStack.size()){
        stream << sumStack.top();
        sumStack.pop();
    }

    return stream.str().size() ? stream.str(): "0";
}

int main()
{
    cout << std::boolalpha;


    try
    {
        cout << "val: " << str << endl;
    } 
    catch(EdxException &e){
        cout << "Excepted..." << endl;
        cout << e.what() << endl;
    }
}