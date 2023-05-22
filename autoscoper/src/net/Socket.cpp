// ----------------------------------
// Copyright (c) 2011, Brown University
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// (1) Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// (2) Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// (3) Neither the name of Brown University nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY BROWN UNIVERSITY “AS IS” WITH NO
// WARRANTIES OR REPRESENTATIONS OF ANY KIND WHATSOEVER EITHER EXPRESS OR
// IMPLIED, INCLUDING WITHOUT LIMITATION ANY WARRANTY OF DESIGN OR
// MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE, EACH OF WHICH ARE
// SPECIFICALLY DISCLAIMED, NOR ANY WARRANTY OR REPRESENTATIONS THAT THE
// SOFTWARE IS ERROR FREE OR THAT THE SOFTWARE WILL NOT INFRINGE ANY
// PATENT, COPYRIGHT, TRADEMARK, OR OTHER THIRD PARTY PROPRIETARY RIGHTS.
// IN NO EVENT SHALL BROWN UNIVERSITY BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
// BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
// OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
// ON ANY THEORY OF LIABILITY OR CAUSE OF ACTION, WHETHER IN CONTRACT,
// STRICT LIABILITY, TORT, NEGLIGENCE OR OTHERWISE, ARISING IN ANY WAY
// OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
// SUCH DAMAGE. ANY RECIPIENT OR USER OF THIS SOFTWARE ACKNOWLEDGES THE
// FOREGOING, AND ACCEPTS ALL RISKS AND LIABILITIES THAT MAY ARISE FROM
// THEIR USE OF THE SOFTWARE.
// ---------------------------------
///\file Socket.cpp
///\author Benjamin Knorlein
///\date 5/14/2018

#ifdef _MSC_VER
#define _CRT_SECURE_NO_WARNINGS
#endif

#include "Socket.h"
#include <iostream>
#include <fstream>
#include "filesystem_compat.hpp"
#include <QTcpSocket>

#define AUTOSCOPER_SOCKET_VERSION_MAJOR 1
#define AUTOSCOPER_SOCKET_VERSION_MINOR 0
#define AUTOSCOPER_SOCKET_VERSION_PATCH 0

Socket::Socket(AutoscoperMainWindow* mainwindow, unsigned long long int listenPort) : m_mainwindow(mainwindow)
{
  tcpServer = new QTcpServer();

  connect(tcpServer, &QTcpServer::newConnection, this, &Socket::onNewConnectionEstablished);
  tcpServer->listen(QHostAddress::LocalHost, listenPort);
}

Socket::~Socket()
{
  for (auto &a : clientConnections){
    a->disconnectFromHost();
  }
}

int constexpr Socket::versionMajor() { return AUTOSCOPER_SOCKET_VERSION_MAJOR; }
int constexpr Socket::versionMinor() { return AUTOSCOPER_SOCKET_VERSION_MINOR; }
int constexpr Socket::versionPatch() { return AUTOSCOPER_SOCKET_VERSION_PATCH; }

QString Socket::versionString()
{
  return QString("%1.%2.%3").arg(QString::number(Socket::versionMajor()), QString::number(Socket::versionMinor()), QString::number(Socket::versionPatch()));
}

void Socket::handleMessage(QTcpSocket * connection, char* data, qint64 length)
{
  unsigned char message_type = data[0];

  switch (message_type)
  {
  case 0:
  {
      // Used for testing socket connection
      connection->write(QByteArray(1, 0));
  }
  break;
  case 1:
    {
      //load trial
      std::string filename = std::string(&data[1],length-1);
      std::ifstream test(filename.c_str());
      if (!test) {
          std::cerr << "Cannot find " << filename.c_str() << std::endl;
          connection->write(QByteArray(1, 0));
      }
      else {
          std::cerr << "load trial " << filename.c_str() << std::endl;
          m_mainwindow->openTrial(QString(filename.c_str()));

          connection->write(QByteArray(1, 1));
      }
    }
    break;
  case 2:
    //load tracking data
    {
      qint32* volume = reinterpret_cast<qint32*>(&data[1]);
      qint32* save_as_matrix = reinterpret_cast<qint32*>(&data[5]);
      qint32* save_as_rows = reinterpret_cast<qint32*>(&data[9]);
      qint32* save_with_commas = reinterpret_cast<qint32*>(&data[13]);
      qint32* convert_to_cm = reinterpret_cast<qint32*>(&data[17]);
      qint32* convert_to_rad = reinterpret_cast<qint32*>(&data[21]);
      qint32* interpolate = reinterpret_cast<qint32*>(&data[25]);
      std::string filename = std::string(&data[29], length - 29);

      if (!std::filesystem::exists(filename)) {
          std::cerr << "Cannot find " << filename.c_str() << std::endl;
          connection->write(QByteArray(1, 0));
      }
      else {
          std::cerr << "load tracking data Volume " << *volume << " : " << filename.c_str() << std::endl;
          std::cerr << "Save as matrix: " << *save_as_matrix << " save as rows: " << *save_as_rows << " save with commas: " << *save_with_commas << " convert to cm: " << *convert_to_cm << " convert to rad: " << *convert_to_rad << " interpolate: " << *interpolate << std::endl;

          m_mainwindow->load_tracking_results(QString(filename.c_str()), *save_as_matrix, *save_as_rows, *save_with_commas, *convert_to_cm, *convert_to_rad, *interpolate, *volume);

          connection->write(QByteArray(1, 2));
      }
    }
    break;
  case 3:
    //save tracking data
    {
      qint32* volume = reinterpret_cast<qint32*>(&data[1]);
      qint32* save_as_matrix = reinterpret_cast<qint32*>(&data[5]);
      qint32* save_as_rows = reinterpret_cast<qint32*>(&data[9]);
      qint32* save_with_commas = reinterpret_cast<qint32*>(&data[13]);
      qint32* convert_to_cm = reinterpret_cast<qint32*>(&data[17]);
      qint32* convert_to_rad = reinterpret_cast<qint32*>(&data[21]);
      qint32* interpolate = reinterpret_cast<qint32*>(&data[25]);
      std::string filename = std::string(&data[29], length - 29);

      // If needed, create parent directory
      bool parent_directory_exists = false;
      if (!std::filesystem::is_directory(filename)) {
          auto parent_directory = std::filesystem::path(filename).parent_path();
          parent_directory_exists = std::filesystem::exists(parent_directory);
          if (!parent_directory_exists){
              std::error_code ec;
              if (std::filesystem::create_directories(parent_directory, ec)) {
                  parent_directory_exists = true;
              }
              else {
                  std::cerr << "save tracking data: failed to create directory " << parent_directory << ": " << ec.message() << std::endl;
              }
          }
      }
      else {
          std::cerr << "save tracking data: failed because filename " << filename << " is a directory" << std::endl;
      }

      // Check permissions
      bool parent_directory_writable = false;
      if (parent_directory_exists){
          auto parent_directory = std::filesystem::path(filename).parent_path();
          std::filesystem::file_status status = std::filesystem::status(parent_directory);
          std::filesystem::perms permissions = status.permissions();
          if ((permissions & (std::filesystem::perms::owner_write
                              | std::filesystem::perms::group_write
                              | std::filesystem::perms::others_write)) != std::filesystem::perms::none) {
              parent_directory_writable = true;
          }
          else {
              std::cerr << "save tracking data: failed because directory " << parent_directory << " is not writable" << std::endl;
          }
      }

      if (!parent_directory_exists || !parent_directory_writable) {
          connection->write(QByteArray(1, 0));
      }
      else {
          std::cerr << "save tracking data Volume " << *volume << " : " << filename.c_str() << std::endl;
          std::cerr << "Save as matrix: " << *save_as_matrix << " save as rows: " << *save_as_rows << " save with commas: " << *save_with_commas << " convert to cm: " << *convert_to_cm << " convert to rad: " << *convert_to_rad << " interpolate: " << *interpolate << std::endl;

          m_mainwindow->save_tracking_results(QString(filename.c_str()), *save_as_matrix, *save_as_rows, *save_with_commas, *convert_to_cm, *convert_to_rad, *interpolate, *volume);

          connection->write(QByteArray(1, 3));
      }
    }
    break;
  case 4:
    //load filter settings
    {
      qint32* camera = reinterpret_cast<qint32*>(&data[1]);
      std::string filename = std::string(&data[5], length - 5);
      if (!std::filesystem::exists(filename)) {
          std::cerr << "Cannot find " << filename.c_str() << std::endl;
          connection->write(QByteArray(1, 0));
      }
      else {

          std::cerr << "load filter settings for camera " << *camera << " : " << filename.c_str() << std::endl;
          m_mainwindow->loadFilterSettings(*camera, QString(filename.c_str()));

          connection->write(QByteArray(1, 4));
      }
    }
    break;
  case 5:
    //set current frame
    {
      qint32* frame = reinterpret_cast<qint32*>(&data[1]);

      std::cerr << "set frame to " << *frame << std::endl;
      m_mainwindow->setFrame(*frame);

      connection->write(QByteArray(1, 5));
    }
    break;
  case 6:
    //get Pose
    {
      qint32* volume = reinterpret_cast<qint32*>(&data[1]);
      qint32* frame = reinterpret_cast<qint32*>(&data[5]);

      std::cerr << "get pose for volume " << *volume << " frame " << *frame << std::endl;
      std::vector<double> pose = m_mainwindow->getPose(*volume,*frame);

      char * ptr = reinterpret_cast<char*>(&pose[0]);
      QByteArray array = QByteArray(1, 6);
      array.append(ptr, sizeof(double) * 6);
      connection->write(array);
    }
    break;
  case 7:
    //set Pose
    {
      qint32* volume = reinterpret_cast<qint32*>(&data[1]);
      qint32* frame = reinterpret_cast<qint32*>(&data[5]);
      double * pose_data = reinterpret_cast<double*>(&data[9]);
      std::vector<double> pose;
      pose.assign(pose_data, pose_data + 6);

      std::cerr << "set pose for volume " << *volume << " frame " << *frame;
      for (auto a : pose)
        std::cerr << " " << a;
      std::cerr << std::endl;
      m_mainwindow->setPose(pose, *volume, *frame);

      connection->write(QByteArray(1, 7));
    }
    break;
  case 8:
    //get NCC
    {
      qint32* volume = reinterpret_cast<qint32*>(&data[1]);
      double * pose_data = reinterpret_cast<double*>(&data[5]);
      std::vector<double> pose;
      pose.assign(pose_data, pose_data + 6);

      std::vector<double> ncc = m_mainwindow->getNCC(*volume, &pose_data[0]);

      //Return double
      char * ptr = reinterpret_cast<char*>(&ncc[0]);
      QByteArray array = QByteArray(1, 8);
      array.append(QByteArray(1, ncc.size()));
      array.append(ptr, sizeof(double) * ncc.size());
      connection->write(array);
    }
  break;
  case 9:
    //set Background
    {
      double * threshold = reinterpret_cast<double*>(&data[1]);

      std::cerr << "set background " << *threshold << std::endl;
      m_mainwindow->setBackground(*threshold);

      connection->write(QByteArray(1, 9));
    }
  break;
  case 10:
    //get the image - cropped
    {
      qint32* volume = reinterpret_cast<qint32*>(&data[1]);
      qint32* camera = reinterpret_cast<qint32*>(&data[5]);
      double * pose_data = reinterpret_cast<double*>(&data[9]);

      std::cerr << "Read images for volume " << *volume << " and camera " << *camera << std::endl;

      std::vector<double> pose;
      pose.assign(pose_data, pose_data + 6);
      unsigned int width, height;
      std::vector<unsigned char> img_Data = m_mainwindow->getImageData(*volume,*camera, &pose_data[0], width, height );

      QByteArray array = QByteArray(1, 10);
      char * ptr = reinterpret_cast<char*>(&width);
      array.append(ptr, sizeof(qint32));
        ptr = reinterpret_cast<char*>(&height);
      array.append(ptr, sizeof(qint32));
      ptr = reinterpret_cast<char*>(&img_Data[0]);
      array.append(ptr, img_Data.size());
      connection->write(array);
      std::cerr << width << " " << height << " " << img_Data.size() << std::endl;

    }
    break;
  case 11:
    //optimize from matlab
    {
      qint32* volumeID = reinterpret_cast<qint32*>(&data[1]);
      qint32* frame = reinterpret_cast<qint32*>(&data[5]);
      qint32* repeats = reinterpret_cast<qint32*>(&data[9]);
      qint32* max_iter = reinterpret_cast<qint32*>(&data[13]);
      double* min_limit = reinterpret_cast<double*>(&data[17]);
      double* max_limit = reinterpret_cast<double*>(&data[25]);
      qint32* stall_iter = reinterpret_cast<qint32*>(&data[33]);

      qint32* dframe = reinterpret_cast<qint32*>(&data[37]);
      qint32* opt_method = reinterpret_cast<qint32*>(&data[41]);
      qint32* cf_model = reinterpret_cast<qint32*>(&data[45]);

      std::cerr << "Running optimization from autoscoper for frame #" << *frame << std::endl;

      m_mainwindow->optimizeFrame(*volumeID, *frame, *dframe, *repeats,
        *opt_method,
        *max_iter, *min_limit, *max_limit,
        *cf_model, *stall_iter);

      connection->write(QByteArray(1, 11));
    }
    break;

  case 12:
    //save full drr image
    {
      std::cerr << "Saving the full DRR image: " << std::endl;

      m_mainwindow->saveFullDRR();

      connection->write(QByteArray(1, 12));
    }
    break;

  case 13:
    // close connection
    {
      std::cerr << "Closing connection to Client..." << std::endl;
      connection->disconnectFromHost();
    }
    break;

  case 14:
    // get number of volumes
    {
      int nvol = m_mainwindow->getNumVolumes();
      QByteArray array = QByteArray(1, 14);
      array.append((char*)&nvol, sizeof(int));
      connection->write(array);
    }
    break;

  case 15:
    // get number of frames
    {
      int nframe = m_mainwindow->getNumFrames();
      QByteArray array = QByteArray(1, 15);
      array.append((char*)&nframe, sizeof(int));
      connection->write(array);
    }
    break;

  case 16:
    // get the version of the server
    {
      QByteArray array = QByteArray(1, 16);
      array.append(versionString().toLocal8Bit());
      connection->write(array);
    }
    break;

  default:
    std::cerr << "Cannot handle message" << std::endl;
    connection->write(QByteArray(1,0));
    break;
  }
}

void Socket::onNewConnectionEstablished()
{
  std::cerr << "New Client is Connected..." << std::endl;
  QTcpSocket *clientConnection = tcpServer->nextPendingConnection();
  connect(clientConnection, &QAbstractSocket::disconnected, this, &Socket::onClientDisconnected);
  connect(clientConnection, &QIODevice::readyRead, this, &Socket::reading);

  clientConnections.push_back(clientConnection);
}

void Socket::onClientDisconnected()
{
  //std::cerr << "client disconnected" << std::endl;
  QTcpSocket * obj = dynamic_cast<QTcpSocket *>(sender());
  if (obj != NULL)
  {
    clientConnections.erase(std::remove(clientConnections.begin(), clientConnections.end(), obj), clientConnections.end());
    obj->deleteLater();
  }
}

void Socket::reading()
{
  QTcpSocket * obj = dynamic_cast<QTcpSocket *>(sender());
  if (obj)
  {
    qint64 avail = obj->bytesAvailable();
    char *data = new char[avail];
    obj->read(data, avail);
    handleMessage(obj, data, avail);
    delete[] data;
  }
}
