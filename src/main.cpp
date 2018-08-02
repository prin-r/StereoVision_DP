#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;

const short regionRadius = 3;
const short mindis = 0;
const short maxdis = 33;
const short maxError = 5000;
const short RtoL = 50;
const short textureThreshold = 1;
const short regionH = (regionRadius<<1)+1;
const short regionW = (regionRadius<<1)+1;

int selectRToL(int col, int scores[], const int &width) {
    short localMax = min(width - regionW, col + maxdis) - col - mindis;
    short indexBest = 0;
    short indexScore = col;
    short scoreBest = scores[col];
    indexScore += width + 1;

    for (short i = 1; i < localMax; i++, indexScore += width + 1) {
        int s = scores[indexScore];

        if (s < scoreBest) {
            scoreBest = s;
            indexBest = i;
        }
    }
    return indexBest;
}

void computeDisparity(short row, int scores[], Mat &disp, const int &width) {
    int indexDis = row*width + regionRadius + mindis;
    short columnScore[width];
    for (short col = mindis; col <= width - regionW; col++) {
        short localMax = 1 + col - mindis - max( 0, col - maxdis + 1);
        short indexScore = col - mindis;
        short bestDis = 0;
        short scoreBest = columnScore[0] = scores[indexScore];
        indexScore += width;

        for (short i = 1; i < localMax; i++, indexScore += width) {
            int s = scores[indexScore];
            columnScore[i] = s;
            if (s < scoreBest) {
                scoreBest = s;
                bestDis = i;
            }
        }

        if (scoreBest > maxError) {
            bestDis = 0;
        } else if (RtoL >= 0) {
            short disRtoL = selectRToL(col - bestDis - mindis, scores, width);

            if (abs(disRtoL - bestDis) > RtoL) bestDis = 0;
        }

        if (textureThreshold > 0 && bestDis > 0 && localMax >= 3) {
            short secondBest = 32765;

            for (short i = 0; i < bestDis - 1; i++) {
                if (columnScore[i] < secondBest) secondBest = columnScore[i];
            }
            for (short i = bestDis + 2; i < localMax; i++) {
                if (columnScore[i] < secondBest) secondBest = columnScore[i];
            }

            int diff = secondBest - scoreBest;

            if ( (diff<<13)+(diff<<10)+(diff<<9)+(diff<<8)+(diff<<4) <= textureThreshold*scoreBest) bestDis = 0;
        }
        disp.data[indexDis++] = (bestDis<<2) + (bestDis<<1) + bestDis;
    }
}

void computeScoreFive(const int top[], const int mid[], const int bottom[], int score[], const short &width) {

    for (int d = mindis; d < maxdis; d++) {
        int indexDst = (d - mindis)*width + (d - mindis);
        int indexSrc = indexDst + regionRadius;
        int indexEnd = indexSrc + (width - d - (regionRadius<<2));

        while (indexSrc < indexEnd) {
            int s = 0;
            int val0 = top[indexSrc - regionRadius];
            int val1 = top[indexSrc + regionRadius];
            int val2 = bottom[indexSrc - regionRadius];
            int val3 = bottom[indexSrc + regionRadius];

            if (val1 < val0) {
                int temp = val0;
                val0 = val1;
                val1 = temp;
            }

            if (val3 < val2) {
                int temp = val2;
                val2 = val3;
                val3 = temp;
            }

            if (val3 < val0) {
                s += val2;
                s += val3;
            } else if (val2 < val1) {
                s += val2;
                s += val0;
            } else {
                s += val0;
                s += val1;
            }

            score[indexDst++] = s + mid[indexSrc++];
        }
    }
}

void computeScoreRow(const short &width, const Mat &imgL, const Mat &imgR, const short &row, int horScore[], short element[]) {

    for (short d = mindis; d < maxdis; d++) {
        short dfm = d - mindis;
        const short colMax = width - d;
        const short scoreMax = colMax - regionW;
        int indexScore = dfm + width*dfm;
        int indexL = width*row + d;
        int indexR = width*row;

        for (short i = 0; i < colMax; i++) {
            short diff = abs( imgL.data[indexL++] - imgR.data[indexR++] );
            element[i] = diff;
        }

        int score = 0;
        for (short i = 0; i < regionW; i++) score += element[i];

        horScore[indexScore++] = score;

        for (short i = 0; i < scoreMax; i++, indexScore++) {
            horScore[indexScore] = score += (element[i+regionW] - element[i]);
        }
    }
}

int main()
{
    Mat imgR, imgL, disp;
    imgR = cv::imread("C:/Users/homuse/Desktop/r.jpg", 0);
    imgL = cv::imread("C:/Users/homuse/Desktop/l.jpg", 0);
    disp  = imgL;

    short w = imgL.cols;
    short h = imgL.rows;
    short activeVer = 0;
    short rangedis = maxdis - mindis;
    short element[w];
    int horizontalDis = (maxdis - mindis)*w;
    int horScore[regionH][horizontalDis];
    int verScore[regionH][horizontalDis];
    int fiveScore[horizontalDis];

    //compute first row

    activeVer = 1;

    for (short row = 0; row < regionH; row++) {
        computeScoreRow(w, imgL, imgR, row, horScore[row], element);
    }

    for (int i = 0; i < horizontalDis; i++) {
        int sum = 0;
        for (int row = 0; row < regionH; row++) {
            sum += horScore[row][i];
        }
        verScore[0][i] = sum;
    }

    //compute remain row

    for (short row = regionH, c = 0; row < h; row++, activeVer++, c++) {
        short oldRow = row%regionH;
        short previous = (activeVer - 1)%regionH;
        short active = activeVer%regionH;

        for (int i = 0; i < horizontalDis; i++) verScore[active][i] = verScore[previous][i] - horScore[oldRow][i];

        computeScoreRow(w, imgL, imgR, row, horScore[oldRow], element);

        for (int i = 0; i < horizontalDis; i++) verScore[active][i] += horScore[oldRow][i];
        if (activeVer >= regionH - 1) {
            computeScoreFive(verScore[(activeVer - (regionRadius<<1))%regionH],verScore[(activeVer - regionRadius)%regionH],verScore[activeVer%regionH], fiveScore, w);
            computeDisparity((row - (regionRadius<<1)), fiveScore, disp, w);
        }
    }

    namedWindow( "Display window1", CV_WINDOW_AUTOSIZE );
    imshow( "Display window1", disp );
    waitKey(0);
    return 0;
}
