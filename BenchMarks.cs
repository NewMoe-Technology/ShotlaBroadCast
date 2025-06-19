using System;
using System.Diagnostics;
using System.Net;
using System.Net.Sockets;

string input_path = @"C:\Users\98246\Downloads\base.mp3";
string output_path = @"C:\Users\98246\Downloads\base_rvc.wav";

UdpClient udpClient = new UdpClient();
IPEndPoint endPoint = new IPEndPoint(IPAddress.Loopback, 8023);

// 这里规约：用\t分隔输入和输出路径
byte[] data = System.Text.Encoding.UTF8.GetBytes(input_path + "\t" + output_path);


// 连接Socket
udpClient.Connect(endPoint);

Stopwatch perf_counter = new Stopwatch();

// 启动性能计时器
perf_counter.Start();

// 发送数据
udpClient.Send(data);

byte[] response = udpClient.Receive(ref endPoint);

perf_counter.Stop();

Console.WriteLine($"Remomte Voice Conversion: {perf_counter.ElapsedMilliseconds} ms");

// 这里规约0为处理成功，1为处理失败
Console.WriteLine("Received response: " + System.Text.Encoding.UTF8.GetString(response));

udpClient.Close();