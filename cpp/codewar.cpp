#include <string>
#include <iostream>
#include <vector>
#include <unordered_set>
#include <queue>
#include <unordered_map>
#include <map>
#include <functional>
#include <algorithm>

using namespace std;

template <typename T, typename... Ts>
bool run_test(function<T(Ts&... args)> &funcToTest, T expected_res, Ts&... args){
    T res = funcToTest(args...);
    return (res == expected_res);
}

typedef pair<int, int> intPair;
int path_finder(std::string maze)
{
    // cout << maze << endl;
    vector<vector<int> > mazeVect{};
    vector<int> row{};
    for (auto el: maze)
    {
        if(el == '\n'){
            mazeVect.push_back(row);
            row.clear();
            continue;
        }
        else{
            row.push_back(el-'0');
        }
    }
    mazeVect.push_back(row);
    size_t mazeFormat = mazeVect.size();

    vector<intPair> possibleMoves{ {0,1}, {0,-1}, {1,0}, {-1, 0} };

    auto addPairs = [](auto &lhs, auto &rhs){ 
        return make_pair(lhs.first + rhs.first, lhs.second + rhs.second); 
    };
    
    auto edgeW = [&mazeVect](auto &lhs, auto &rhs){ 
        return abs(mazeVect[lhs.first][lhs.second] - mazeVect[rhs.first][rhs.second]); 
    };

    auto nodeOutOfBound = [&mazeFormat](auto &node){ 
        return (   node.first < 0 
                || node.first > mazeFormat-1
                || node.second < 0 
                || node.second > mazeFormat-1);
    };

    struct HashPair
    {
        size_t operator()(const intPair &lhs) const 
        {
            size_t fHash = hash<int>()(lhs.first);
            size_t sHash = hash<int>()(lhs.second);
            return fHash ^ sHash;
        }
    };
    

    unordered_set<intPair, HashPair> visVertex{};
    auto get_childs = [&](intPair &node){
        vector<intPair> childs{};
        for (auto el: possibleMoves){
            auto nextNode = addPairs(el, node);

            if ( nodeOutOfBound(nextNode) || visVertex.find(nextNode) != visVertex.end())
                continue;
            childs.push_back(nextNode);
        }
        return childs;
    };
    function<bool(const pair<intPair, int>&, const pair<intPair, int>&)> comp = 
    [](const pair<intPair, int>& v1, const pair<intPair, int>& v2){
        return (v1.second < v2.second);
    };

    map<intPair, int> edgeWeightTrack{};
    unordered_map<intPair, int, HashPair> shortesDict{};
    intPair startVer = {0, 0};

    edgeWeightTrack[startVer] = 0;
    queue<intPair> pathQueue;
    pathQueue.push(startVer);

    while (pathQueue.size())
    {
        auto tmpNode = pathQueue.front();
        pathQueue.pop();
        visVertex.insert(tmpNode);
        auto childs = get_childs(tmpNode);

        for (auto child: childs)
        {
            if( visVertex.find(child) != visVertex.end() )
                continue;

            auto childWeight = edgeWeightTrack[tmpNode] + edgeW(child, tmpNode);
            if (edgeWeightTrack.find(child) == edgeWeightTrack.end()) {
                edgeWeightTrack[child] = childWeight;
                shortesDict[child] = childWeight;
            }
            else {
                if(edgeWeightTrack[child] > childWeight){
                    edgeWeightTrack[child] = childWeight;
                    shortesDict[child] = childWeight;
                }
            }
        }
        // cout << "===========================" << endl;
        edgeWeightTrack.erase(tmpNode);
        // cout << "sizeof edgeWeightTrack: " << edgeWeightTrack.size() << endl;
        if(edgeWeightTrack.empty())
            break;
        auto minWeighetNode = 
            min_element(edgeWeightTrack.begin(), edgeWeightTrack.end(), comp);
        pathQueue.push(minWeighetNode->first);
    }


    if (shortesDict.find({mazeFormat-1, mazeFormat-1}) != shortesDict.end())
    {
        cout << "And the shortest weight is: " << shortesDict[{mazeFormat-1, mazeFormat-1}] << endl;
        return shortesDict[{mazeFormat-1, mazeFormat-1}];
    }

    return -1;
}

template<typename T>
struct MyNode {
    MyNode(T val): 
        value(val),
        left{nullptr},
        right{nullptr},
        root{nullptr} {}
    T value;
    MyNode* left;
    MyNode* right;
    MyNode* root;
};

template<typename T>
class BST
{
    MyNode<T> *rootNode{nullptr};
    int levelNbr{};
public:
    BST(MyNode<T> *root): rootNode(root){}
    ~BST() {}

    void insert(MyNode<T> *node) {
        return _insert(node, rootNode);
    }

    friend ostream &operator<<(ostream &os, BST<T> &bst){
        vector<T> vals{};
        bst._postOrderTraversal(bst.rootNode, vals);
        for(auto &el: vals)
            os << el << ", ";

        return os;
   }

private:
    void _preOrderTraversal(const MyNode<T> *node, vector<T> &vals) {
        if(!node) return;
        vals.push_back(node->value);
        _preOrderTraversal(node->left, vals);
        _preOrderTraversal(node->right, vals);
    }

    void _inOrderTraversal(const MyNode<T> *node, vector<T> &vals) {
        if(!node) return;
        _inOrderTraversal(node->left, vals);
        vals.push_back(node->value);
        _inOrderTraversal(node->right, vals);
    }

    void _postOrderTraversal(const MyNode<T> *node, vector<T> &vals) {
        if(!node) return;
        _postOrderTraversal(node->left, vals);
        _postOrderTraversal(node->right, vals);
        vals.push_back(node->value);
    }

    void _insert(MyNode<T> *node, MyNode<T> *refNode)
    {
        if(node->value > refNode->value)
        {
            if(refNode->right)
                return _insert(node, refNode->right);
            
            refNode->right = node;
            node->root = refNode;
        }
        else if(node->value < refNode->value)
        {
            if(refNode->left)
                return _insert(node, refNode->left);
            
            refNode->left = node;
            node->root = refNode;
        }
        else
        {
            node->left = refNode->left;
            refNode->left = node;
            node->root = refNode;
        }
    }
};

int main() 
{
    cout << std::boolalpha;
    // std::string s7 = "000000\n""000000\n""000000\n""000010\n""000109\n""001010";
    // function<int(string&)> func = path_finder;
    // cout << run_test(func, 4, s7) << endl;
    vector<int> vecArr;
    vecArr = {10, 5, 1, 7, 40, 50};
    sort(vecArr.begin(), vecArr.end());
    for (auto el: vecArr)
        cout << el << ", ";
    cout << endl;

    int mid = vecArr.size() / 2;
    int left = mid-1;
    int right = mid+1;
    MyNode<int>* rootNode = new MyNode<int>(vecArr.at(mid));
    BST<int> binSearchTree = BST<int>(rootNode);
    cout << "Add mid: " << vecArr.at(mid) << endl;

    while(left >= 0 || right < vecArr.size())
    {
        if(left >= 0)
        {
            cout << "Add left: " << vecArr.at(left) << endl;
            MyNode<int>* leftNode = new MyNode<int>(vecArr.at(left));
            binSearchTree.insert(leftNode);
            left--;
        }
        if(right < vecArr.size())
        {
            cout << "Add right: " << vecArr.at(right) << endl;
            MyNode<int>* rightNode = new MyNode<int>(vecArr.at(right));
            binSearchTree.insert(rightNode);
            right++;
        }
    }
    cout << "end" << endl;
    cout << binSearchTree << endl;

    return 0;
}