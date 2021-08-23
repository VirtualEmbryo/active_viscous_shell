h=0.1;
h2 = 0.04;

Point(1) = {0,0,0,h};
Point(2) = {1,0,0,h2};
Point(3) = {0,1,0,h};
Point(4) = {0,0,1,h2};

Circle(1) = {2,1,3};
Circle(8) = {4,1,2};
Circle(11) = {3,1,4};

Line Loop(1) = {1,11,8};
#Line Loop(2) = {2,7,-11};
#Line Loop(3) = {3,-12,-7};
#Line Loop(4) = {4,-8,12};


Ruled Surface(1) = {1};


Volume (1) = {1};

Physical Surface(1) = {1};
//+
Physical Curve(12) = {1};
//+
Physical Curve(13) = {8};
//+
Physical Curve(14) = {11};
