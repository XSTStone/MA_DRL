import time
import pandas as pd
import math
import numpy as np
import geopy.distance
import matplotlib.pyplot as plt
import matplotlib
import simpy
import numba
import networkx as nx
from PIL import Image
from scipy.optimize import linear_sum_assignment
import os

###############################################################################
#################################    Simpy    #################################
###############################################################################

receivedDataBlocks = []
createdBlocks = []
seed = np.random.seed(1)

upGSLRates = []
downGSLRates = []
interRates = []
intraRate = []

'''
    General Block transmission stats
    模拟结果统计函数,用于计算数据块在卫星网络中的传输性能指标(延迟、跳数等),并输出关键统计结果
    最终返回一个封装了所有统计信息的 Results 对象
    '''
def getBlockTransmissionStats(timeToSim, GTs, constellationType):
    allTransmissionTimes = []  # 存储所有数据块的总传输时间
    largestTransmissionTime = (0, None)  # 记录最长传输时间(值, 对应的块)
    mostHops = (0, None)  # 记录最多跳数(值, 对应的块)
    queueLat = []  # 存储每个块的队列延迟(排队时间)
    txLat = []  # 存储每个块的传输延迟(数据发送时间)
    propLat = []  # 存储每个块的传播延迟(信号在空间传播的时间)
    latencies = [queueLat, txLat, propLat]  # 延迟分类列表(未直接使用)
    blocks = []  # 预留存储完成传输的块(当前代码未填充)

    for block in receivedDataBlocks:
        time = block.getTotalTransmissionTime()  # 获取块的总传输时间(方法未展示)
        hops = len(block.checkPoints)  # 跳数(通过检查点数量计算,checkPoints记录每跳的接收时间)

        # 更新最长传输时间记录
        if largestTransmissionTime[0] < time:
            largestTransmissionTime = (time, block)

        # 更新最多跳数记录
        if mostHops[0] < hops:
            mostHops = (hops, block)

        allTransmissionTimes.append(time)  # 记录总传输时间

        # 记录三种延迟(队列、传输、传播)
        queueLat.append(block.getQueueTime()[0])  # 队列延迟(方法未展示)
        txLat.append(block.txLatency)  # 传输延迟(块属性)
        propLat.append(block.propLatency)  # 传播延迟(块属性)

    avgTime = np.mean(allTransmissionTimes)  # 平均传输时间(使用numpy计算均值)
    totalTime = sum(allTransmissionTimes)  # 所有块的总传输时间之和(用于计算延迟占比) 

    print("\n########## Results #########\n")
    print(f"The simulation took {timeToSim} seconds to run")  # 模拟总耗时
    print(f"A total of {len(createdBlocks)} data blocks were created")  # 总创建的块数(全局变量)
    print(f"A total of {len(receivedDataBlocks)} data blocks were transmitted")  # 成功传输的块数
    print(f"A total of {len(createdBlocks) - len(receivedDataBlocks)} data blocks were stuck")  # 未传输的块数(卡住的块)
    print(f"Average transmission time for all blocks were {avgTime}")  # 平均传输时间
    print('Total latecies:\nQueue time: {}%\nTransmission time: {}%\nPropagation time: {}%'.format(
        '%.4f' % float(sum(queueLat)/totalTime*100),  # 队列延迟占比
        '%.4f' % float(sum(txLat)/totalTime*100),     # 传输延迟占比
        '%.4f' % float(sum(propLat)/totalTime*100)    # 传播延迟占比
    ))

    results = Results(
        finishedBlocks=blocks,  # 完成传输的块(当前为空)
        constellation=constellationType,  # 星座类型
        GTs=GTs,  # 地面站信息
        meanTotalLatency=avgTime,  # 平均总延迟
        meanQueueLatency=np.mean(queueLat),  # 平均队列延迟
        meanPropLatency=np.mean(propLat),    # 平均传播延迟
        meanTransLatency=np.mean(txLat),     # 平均传输延迟
        perQueueLatency=sum(queueLat)/totalTime*100,  # 队列延迟占比(%)
        perPropLatency=sum(propLat)/totalTime*100,    # 传播延迟占比(%)
        perTransLatency=sum(txLat)/totalTime*100       # 传输延迟占比(%)
    )

    return results

'''
    模拟进度显示函数
    用于实时显示模拟的进度,包括已完成的模拟时间、估计的剩余时间和当前模拟时间
    每经过 timeStepSize 秒,进度将更新并显示
'''
def simProgress(simTimelimit, env):
    timeSteps = 100
    timeStepSize = simTimelimit/timeSteps
    progress = 1
    startTime = time.time()
    yield env.timeout(timeStepSize)
    while True:
        elapsedTime = time.time() - startTime
        estimatedTimeRemaining = elapsedTime * (timeSteps/progress) - elapsedTime
        print("Simulation progress: {}% Estimated time remaining: {} seconds Current simulation time: {}".format(progress, int(estimatedTimeRemaining), env.now), end='\r')
        yield env.timeout(timeStepSize)
        progress += 1


###############################################################################
###############################    Constants    ###############################
###############################################################################
'''
    物理常量定义
    包括地球半径、重力加速度、地球质量、地球自转周期、光速、玻尔兹曼常数和有效天线效率等
    这些常量在模拟中用于物理计算和模型假设
'''
Re  = 6378e3            # Radius of the earth [m]
G    = 6.67259e-11      # Universal gravitational constant [m^3/kg s^2]
Me  = 5.9736e24         # Mass of the earth
Te  = 86164.28450576939 # Time required by Earth for 1 rotation
Vc  = 299792458         # Speed of light [m/s]
k   = 1.38e-23          # Boltzmann's constant
eff = 0.55              # Efficiency of the parabolic antenna

###############################################################################
###############################     Classes    ################################
###############################################################################


class Results:
    def __init__(self, finishedBlocks, constellation, GTs, meanTotalLatency, meanQueueLatency, meanTransLatency, meanPropLatency, perQueueLatency, perPropLatency,perTransLatency):

        self.GTs = GTs
        self.finishedBlocks = finishedBlocks
        self.constellation = constellation
        self.meanTotalLatency = meanTotalLatency
        self.meanQueueLatency = meanQueueLatency
        self.meanPropLatency = meanPropLatency
        self.meanTransLatency = meanTransLatency
        self.perQueueLatency = perQueueLatency
        self.perPropLatency = perPropLatency
        self.perTransLatency = perTransLatency

'''
    用于数据块序列化的辅助类
    提取原始数据块的关键属性,生成轻量可序列化的对象
'''
class BlocksForPickle:
    def __init__(self, block):
        self.size = 64800  # size in bits
        self.ID = block.ID  # a string which holds the source id, destination id, and index of the block, e.g. "1_2_12"
        self.timeAtFull = block.timeAtFull  # the simulation time at which the block was full and was ready to be sent.
        self.creationTime = block.creationTime  # the simulation time at which the block was created.
        self.timeAtFirstTransmission = block.timeAtFirstTransmission  # the simulation time at which the block left the GT.
        self.checkPoints = block.checkPoints  # list of simulation reception times at node with the first entry being the reception time at first sat - can be expanded to include the sat IDs at each checkpoint
        self.checkPointsSend = block.checkPointsSend  # list of times after the block was sent at each node
        self.path = block.path
        self.queueLatency = block.queueLatency  # total time acumulated in the queues
        self.txLatency = block.txLatency  # total transmission time
        self.propLatency = block.propLatency  # total propagation latency
        self.totLatency = block.totLatency  # total latency


'''
    用于建模卫星通信中的射频(Radio Frequency, RF)链路
    封装了射频链路的关键参数(如频率、带宽、天线增益等)
    并通过初始化方法计算链路的核心性能指标(如总增益、噪声功率、增益与系统温度比等)
    目的是为卫星网络模拟提供物理层的射频通信参数支持,用于后续计算数据传输速率、延迟等性能
'''
class RFlink:
    def __init__(self, frequency, bandwidth, maxPtx, aDiameterTx, aDiameterRx, pointingLoss, noiseFigure,
                 noiseTemperature, min_rate):
        self.f = frequency
        self.B = bandwidth
        self.maxPtx = maxPtx
        self.maxPtx_db = 10 * math.log10(self.maxPtx)
        self.Gtx = 10 * math.log10(eff * ((math.pi * aDiameterTx * self.f / Vc) ** 2))
        self.Grx = 10 * math.log10(eff * ((math.pi * aDiameterRx * self.f / Vc) ** 2))
        self.G = self.Gtx + self.Grx - 2 * pointingLoss
        self.No = 10 * math.log10(self.B * k) + noiseFigure + 10 * math.log10(
            290 + (noiseTemperature - 290) * (10 ** (-noiseFigure / 10)))
        self.GoT = 10 * math.log10(eff * ((math.pi * aDiameterRx * self.f / Vc) ** 2)) - noiseFigure - 10 * math.log10(
            290 + (noiseTemperature - 290) * (10 ** (-noiseFigure / 10)))
        self.min_rate = min_rate

    def __repr__(self):
        return '\n Carrier frequency = {} GHz\n Bandwidth = {} MHz\n Transmission power = {} W\n Gain per antenna: Tx {}  Rx {}\n Total antenna gain = {} dB\n Noise power = {} dBW\n G/T = {} dB/K'.format(
            self.f / 1e9,
            self.B / 1e6,
            self.maxPtx,
            '%.2f' % self.Gtx,
            '%.2f' % self.Grx,
            '%.2f' % self.G,
            '%.2f' % self.No,
            '%.2f' % self.GoT,
        )

'''
    全称为 Free Space Optics Link自由空间光通信链路
    用于建模卫星通信中的自由空间光通信链路的关键参数
    与射频RF通信互补
    通常具有更高的数据速率和更小的设备重量
'''
class FSOlink:
    def __init__(self, data_rate, power, comm_range, weight):
        self.data_rate = data_rate
        self.power = power
        self.comm_range = comm_range
        self.weight = weight

    def __repr__(self):
        return '\n Data rate = {} Mbps\n Power = {} W\n Transmission range = {} km\n Weight = {} kg'.format(
            self.data_rate / 1e6,
            self.power,
            self.comm_range / 1e3,
            self.weight)


'''
    用于建模卫星星座中的单个轨道平面
    封装了轨道平面的核心参数
    提供了轨道动态调整(旋转)的方法
    其核心作用是模拟卫星在轨道平面中的分布、运动规律,以及轨道随时间的位置变化,为卫星网络的通信和路由模拟提供基础支撑。
'''
class OrbitalPlane:
    def __init__(self, ID, h, longitude, inclination, n_sat, min_elev, firstID, env):
        self.ID = ID 								# A unique ID given to every orbital plane = index in Orbital_planes, string 轨道平面的唯一标识(字符串类型,通常为轨道平面在星座中的索引)
        self.h = h									# Altitude of deployment 轨道高度(单位：米),即卫星部署的海拔高度。
        self.longitude = longitude					# Longitude angle where is intersects equator [radians] 轨道平面与赤道交点的初始经度
        self.inclination = math.pi/2 - inclination	# Inclination of the orbit form [radians] 轨道平面的倾角,通常为90度减去轨道的倾角
        self.n_sat = n_sat							# Number of satellites in plane 轨道平面上卫星的数量
        self.period = 2 * math.pi * math.sqrt((self.h+Re)**3/(G*Me))	# Orbital period of the satellites in seconds 轨道周期,用于计算卫星的运动周期
        self.v = 2*math.pi * (h + Re) / self.period						# Orbital velocity of the satellites in m/s 轨道速度,用于计算卫星的运动速度
        self.min_elev = math.radians(min_elev)							# Minimum elevation angle for ground comm.  最小仰角,用于限制地面站与卫星的可视角度
        self.max_alpha = math.acos(Re*math.cos(self.min_elev)/(self.h+Re))-self.min_elev	# Maximum angle at the center of the Earth w.r.t. yaw  几何角度参数,用于计算卫星覆盖地面的最大距离
        self.max_beta  = math.pi/2-self.max_alpha-self.min_elev								# Maximum angle at the satellite w.r.t. yaw  几何角度参数,用于计算卫星覆盖地面的最大距离
        self.max_distance_2_ground = Re*math.sin(self.max_alpha)/math.sin(self.max_beta)	# Maximum distance to a servable ground station 卫星到可服务地面站的最大距离
        
        # Adding satellites
        self.first_sat_ID = firstID # Unique ID of the first satellite in the orbital plane 轨道平面上第一个卫星的唯一标识
        
        self.sats = []              # List of satellites in the orbital plane 轨道平面上的卫星列表 
        for i in range(n_sat):      # 卫星ID由 firstID 拼接索引生成
            self.sats.append(Satellite(self.first_sat_ID + str(i), int(self.ID), int(i), self.h, self.longitude, self.inclination, self.n_sat, env))
        
        self.last_sat_ID = self.first_sat_ID + str(len(self.sats) - 1) # Unique ID of the last satellite in the orbital plane 轨道平面上最后一个卫星的唯一标识
    
    '''
        输出轨道平面的信息,包括轨道平面ID、轨道高度、经度、倾角、卫星数量、周期和速度
    '''
    def __repr__(self):             
        return '\nID = {}\n altitude= {} km\n longitude= {} deg\n inclination= {} deg\n number of satellites= {}\n period= {} hours\n satellite speed= {} km/s'.format(
            self.ID,
            self.h/1e3,
            '%.2f' % math.degrees(self.longitude),
            '%.2f' % math.degrees(self.inclination),
            '%.2f' % self.n_sat,
            '%.2f' % (self.period/3600),
            '%.2f' % (self.v/1e3))

    '''
        用于模拟 地球自转对轨道平面位置的影响 ,通过调整轨道平面的经度( longitude )实现轨道的动态旋转
    '''
    def rotate(self, delta_t):
        """
        Rotates the orbit according to the elapsed time by adjusting the longitude. The amount the longitude is adjusted
        is based on the fraction the elapsed time makes up of the time it takes the Earth to complete a full rotation.
        """

        # Change in longitude and phi due to Earth's rotation
        self.longitude = self.longitude + 2*math.pi*delta_t/Te 
        self.longitude = self.longitude % (2*math.pi)
        # Rotating every satellite in the orbital plane
        for sat in self.sats:
            sat.rotate(delta_t, self.longitude, self.period)


'''
    建模单个卫星的完整行为,包括卫星的物理属性(位置、轨道参数)、通信链路管理(与地面站、其他卫星的连接)、数据块传输逻辑(接收、发送、延迟计算),以及动态轨道运动模拟(随时间旋转)
    核心作用是为卫星网络的路由算法、性能评估(如延迟、吞吐量)提供基础的卫星实例支持
'''
class Satellite:
    def __init__(self, ID, in_plane, i_in_plane, h, longitude, inclination, n_sat, env, quota = 500, power = 10):
        self.ID = ID                    # A unique ID given to every satellite
        self.in_plane = in_plane        # Orbital plane where the satellite is deployed
        self.i_in_plane = i_in_plane    # Index in orbital plane
        self.quota = quota              # Quota of the satellite
        self.h = h                      # Altitude of deployment
        self.power = power              # Transmission power
        self.minElevationAngle = 30     # Value is taken from NGSO constellation design chapter.

        # Spherical Coordinates before inclination (r,theta,phi)
        self.r = Re+self.h 
        self.theta = 2 * math.pi * self.i_in_plane / n_sat
        self.phi = longitude
        
        # Inclination of the orbital plane
        self.inclination = inclination
        
        # Cartesian coordinates  (x,y,z)
        self.x = self.r * (math.sin(self.theta)*math.cos(self.phi) - math.cos(self.theta)*math.sin(self.phi)*math.sin(self.inclination))
        self.y = self.r * (math.sin(self.theta)*math.sin(self.phi) + math.cos(self.theta)*math.cos(self.phi)*math.sin(self.inclination))
        self.z = self.r * math.cos(self.theta)*math.cos(self.inclination)
        
        self.polar_angle = self.theta               # Angle within orbital plane [radians]
        self.latitude = math.asin(self.z/self.r)   # latitude corresponding to the satellite
        # longitude corresponding to satellite
        if self.x > 0:
            self.longitude = math.atan(self.y/self.x)
        elif self.x < 0 and self.y >= 0:
            self.longitude = math.pi + math.atan(self.y/self.x)
        elif self.x < 0 and self.y < 0:
            self.longitude = math.atan(self.y/self.x) - math.pi
        elif self.y > 0:
            self.longitude = math.pi/2
        elif self.y < 0:
            self.longitude = -math.pi/2
        else:
            self.longitude = 0

        self.waiting_list = {}
        self.applications = []
        self.n_sat = n_sat

        # downlink params
        f = 20e9  # Carrier frequency GEO to ground (Hz)
        B = 500e6  # Maximum bandwidth
        maxPtx = 10  # Maximum transmission power in W
        Adtx = 0.26  # Transmitter antenna diameter in m
        Adrx = 0.33  # Receiver antenna diameter in m
        pL = 0.3  # Pointing loss in dB
        Nf = 1.5  # Noise figure in dB
        Tn = 50  # Noise temperature in K
        min_rate = 10e3  # Minimum rate in kbps
        self.ngeo2gt = RFlink(f, B, maxPtx, Adtx, Adrx, pL, Nf, Tn, min_rate)
        self.downRate = 0

        # simpy
        self.env = env
        self.sendBufferGT = ([env.event()], [])  # ([self.env.event()], [DataBlock(0, 0, "0", 0)])
        self.sendBlocksGT = []  # env.process(self.sendBlock())  # simpy processes which send the data blocks
        self.sats = []
        self.linkedGT = None
        self.GTDist = None
        # list of data blocks waiting on their propagation delay.
        self.tempBlocks = []  # This list is used to so the block can have their paths changed when the constellation is moved

        self.intraSats = []
        self.interSats = []
        self.sendBufferSatsIntra = []
        self.sendBufferSatsInter = []
        self.sendBlocksSatsIntra = []
        self.sendBlocksSatsInter = []
        self.newBuffer = [False]

    def maxSlantRange(self):
        """
        Maximum distance from satellite to edge of coverage area is calculated using the following formula:
        D_max(minElevationAngle, h) = sqrt(Re**2*sin**2(minElevationAngle) + 2*Re*h + h**2) - Re*sin(minElevationAngle)
        This formula is based on the NGSO constellation design chapter page 16.
        """
        eps = math.radians(self.minElevationAngle)

        distance = math.sqrt((Re+self.h)**2-(Re*math.cos(eps))**2) - Re*math.sin(eps)

        return distance

    def __repr__(self):
        return '\nID = {}\n orbital plane= {}, index in plane= {}, h={}\n pos r = {}, pos theta = {},' \
               ' pos phi = {},\n pos x= {}, pos y= {}, pos z= {}\n inclination = {}\n polar angle = {}' \
               '\n latitude = {}\n longitude = {}'.format(
                self.ID,
                self.in_plane,
                self.i_in_plane,
                '%.2f' % self.h,
                '%.2f' % self.r,
                '%.2f' % self.theta,
                '%.2f' % self.phi,
                '%.2f' % self.x,
                '%.2f' % self.y,
                '%.2f' % self.z,
                '%.2f' % math.degrees(self.inclination),
                '%.2f' % math.degrees(self.polar_angle),
                '%.2f' % math.degrees(self.latitude),
                '%.2f' % math.degrees(self.longitude))

    def createReceiveBlockProcess(self, block, propTime):
        """
        Function which starts a receiveBlock process upon receiving a block from a transmitter.
        """
        process = self.env.process(self.receiveBlock(block, propTime))

    def receiveBlock(self, block, propTime):
        """
        Simpy process function:

        This function is used to handle the propagation delay of data blocks. This is done simply by waiting the time
        of the propagation delay and adding the block to the send-buffer afterwards. Since there are multiple buffers,
        this function looks at the next step in the blocks path and adds the block to the correct send-buffer.

        While the transmission delay is handled at the transmitter, the transmitter cannot also wait for the propagation
        delay, otherwise the send-buffer might be overfilled.

        Using this structure, if there are to be implemented limits on the sizes of the "receive-buffer" it could be
        handled by either limiting the amount of these processes that can occur at the same time, or limiting the size
        of the send-buffer.

        Adds the propagation time to the block attribute
        """
        # wait for block to fully propagate
        self.tempBlocks.append(block)

        yield self.env.timeout(propTime)

        if block.path == -1:
            return

        # ANCHOR KPI: propLatency receive block from sat
        block.propLatency += propTime

        for i, tempBlock in enumerate(self.tempBlocks):
            if block.ID == tempBlock.ID:
                self.tempBlocks.pop(i)
                break

        block.checkPoints.append(self.env.now)

        # get this satellites index in the blocks path
        index = None
        for i, step in enumerate(block.path):
            if self.ID == step[0]:
                index = i

        # check if next step in path is GT (last step in path)
        if index == len(block.path) - 2:
            # add block to GT send-buffer
            if not self.sendBufferGT[0][0].triggered:
                self.sendBufferGT[0][0].succeed()
                self.sendBufferGT[1].append(block)
            else:
                newEvent = self.env.event().succeed()
                self.sendBufferGT[0].append(newEvent)
                self.sendBufferGT[1].append(block)

        else:
            ID = None
            isIntra = False
            # get ID of next sat
            for sat in self.intraSats:
                id = sat[1].ID
                if id == block.path[index + 1][0]:
                    ID = sat[1].ID
                    isIntra = True
            for sat in self.interSats:
                id = sat[1].ID
                if id == block.path[index + 1][0]:
                    ID = sat[1].ID

            if ID is not None:
                sendBuffer = None
                # find send-buffer for the satellite
                if isIntra:
                    for buffer in self.sendBufferSatsIntra:
                        if ID == buffer[2]:
                            sendBuffer = buffer
                else:
                    for buffer in self.sendBufferSatsInter:
                        if ID == buffer[2]:
                            sendBuffer = buffer

                # add block to buffer
                if not sendBuffer[0][0].triggered:
                    sendBuffer[0][0].succeed()
                    sendBuffer[1].append(block)
                else:
                    newEvent = self.env.event().succeed()
                    sendBuffer[0].append(newEvent)
                    sendBuffer[1].append(block)

            else:
                print(
                    "ERROR! Sat {} tried to send block to {} but did not have it in its linked satellite list".format(
                        self.ID, block.path[index + 1][0]))
                print(block.path)
                for neighbor in self.interSats:
                    print(neighbor[1].ID)
                for neighbor in self.intraSats:
                    print(neighbor[1].ID)
                print(block.isNewPath)
                print(block.oldPath)
                print(block.newPath)
                exit()

    def sendBlock(self, destination, isSat, isIntra = None):
        """
        Simpy process function:

        Sends data blocks that are filled and added to one of the send-buffers, a buffer which consists of a list of
        events and data blocks. Since there are multiple send-buffers, the function finds the correct buffer given
        information regarding the desired destination satellite or GT. The function monitors the send-buffer, and when
        the buffer contains one or more triggered events, the function will calculate the time it will take to send the
        block and trigger an event which notifies a separate process that a block has been sent.

        A process is running this method for each ISL and for the downLink GSL the satellite has. This will usually be
        4 ISL processes and 1 GSL process.
        """

        if isIntra is not None:
            sendBuffer = None
            if isSat:
                if isIntra:
                    for buffer in self.sendBufferSatsIntra:
                        if buffer[2] == destination[1].ID:
                            sendBuffer = buffer
                else:
                    for buffer in self.sendBufferSatsInter:
                        if buffer[2] == destination[1].ID:
                            sendBuffer = buffer
        else:
            sendBuffer = self.sendBufferGT

        while True:
            try:
                yield sendBuffer[0][0]

                # ANCHOR KPI: queueLatency at sat
                sendBuffer[1][0].checkPointsSend.append(self.env.now)

                if isSat:
                    timeToSend = sendBuffer[1][0].size / destination[2]

                    propTime = self.timeToSend(destination)
                    yield self.env.timeout(timeToSend)

                    receiver = destination[1]

                else:
                    propTime = self.timeToSend(self.linkedGT.linkedSat)
                    timeToSend = sendBuffer[1][0].size / self.downRate
                    yield self.env.timeout(timeToSend)

                    receiver = self.linkedGT

                # When the constellations move, the only case where this process can simply continue, is when the
                # receiver is the same, and there is a block already ready to be sent. The only place where the process
                # can continue from, is as a result right here. Furthermore, the only processes this can happen for are
                # the inter-ISL processes.
                # Due to having to remake buffers when the satellites move, it is necessary for the process to "find"
                # the correct buffer again - the process uses a reference to the buffer: "sendBuffer".
                # To avoid remaking the reference every time a block is sent, the list of boolean values: self.newBuffer
                # is used to indicate when the constellation is moved,

                if True in self.newBuffer and not isIntra and isSat: # remake reference to buffer
                    if isIntra is not None:
                        sendBuffer = None
                        if isSat:
                            if isIntra:
                                for buffer in self.sendBufferSatsIntra:
                                    if buffer[2] == destination[1].ID:
                                        sendBuffer = buffer
                            else:
                                for buffer in self.sendBufferSatsInter:
                                    if buffer[2] == destination[1].ID:
                                        sendBuffer = buffer
                    else:
                        sendBuffer = self.sendBufferGT

                    for index, val in enumerate(self.newBuffer):
                        # each process will one by one remake their reference, and change one value to True.
                        # After all processes has done this, all values are back to False
                        if val:
                            self.newBuffer[index] = False
                            break

                # ANCHOR KPI: txLatency ISL
                sendBuffer[1][0].txLatency += timeToSend
                receiver.createReceiveBlockProcess(sendBuffer[1][0], propTime)

                # remove from own buffer
                if len(sendBuffer[0]) == 1:
                    sendBuffer[0].pop(0)
                    sendBuffer[1].pop(0)
                    sendBuffer[0].append(self.env.event())

                else:
                    sendBuffer[0].pop(0)
                    sendBuffer[1].pop(0)
            except simpy.Interrupt:
                break

    def adjustDownRate(self):

        speff_thresholds = np.array(
            [0, 0.434841, 0.490243, 0.567805, 0.656448, 0.789412, 0.889135, 0.988858, 1.088581, 1.188304, 1.322253,
             1.487473, 1.587196, 1.647211, 1.713601, 1.779991, 1.972253, 2.10485, 2.193247, 2.370043, 2.458441,
             2.524739, 2.635236, 2.637201, 2.745734, 2.856231, 2.966728, 3.077225, 3.165623, 3.289502, 3.300184,
             3.510192, 3.620536, 3.703295, 3.841226, 3.951571, 4.206428, 4.338659, 4.603122, 4.735354, 4.933701,
             5.06569, 5.241514, 5.417338, 5.593162, 5.768987, 5.900855])
        lin_thresholds = np.array(
            [1e-10, 0.5188000389, 0.5821032178, 0.6266138647, 0.751622894, 0.9332543008, 1.051961874, 1.258925412,
             1.396368361, 1.671090614, 2.041737945, 2.529297996, 2.937649652, 2.971666032, 3.25836701, 3.548133892,
             3.953666201, 4.518559444, 4.83058802, 5.508076964, 6.45654229, 6.886522963, 6.966265141, 7.888601176,
             8.452788452, 9.354056741, 10.49542429, 11.61448614, 12.67651866, 12.88249552, 14.48771854, 14.96235656,
             16.48162392, 18.74994508, 20.18366364, 23.1206479, 25.00345362, 30.26913428, 35.2370871, 38.63669771,
             45.18559444, 49.88844875, 52.96634439, 64.5654229, 72.27698036, 76.55966069, 90.57326009])
        db_thresholds = np.array(
            [-100.00000, -2.85000, -2.35000, -2.03000, -1.24000, -0.30000, 0.22000, 1.00000, 1.45000, 2.23000, 3.10000,
             4.03000, 4.68000, 4.73000, 5.13000, 5.50000, 5.97000, 6.55000, 6.84000, 7.41000, 8.10000, 8.38000, 8.43000,
             8.97000, 9.27000, 9.71000, 10.21000, 10.65000, 11.03000, 11.10000, 11.61000, 11.75000, 12.17000, 12.73000,
             13.05000, 13.64000, 13.98000, 14.81000, 15.47000, 15.87000, 16.55000, 16.98000, 17.24000, 18.10000,
             18.59000, 18.84000, 19.57000])

        pathLoss = 10*np.log10((4*math.pi*self.linkedGT.linkedSat[0]*self.ngeo2gt.f/Vc)**2)
        snr = 10**((self.ngeo2gt.maxPtx_db + self.ngeo2gt.G - pathLoss - self.ngeo2gt.No)/10)
        shannonRate = self.ngeo2gt.B*np.log2(1+snr)

        feasible_speffs = speff_thresholds[np.nonzero(lin_thresholds <= snr)]
        speff = self.ngeo2gt.B * feasible_speffs[-1]

        self.downRate = speff

    def timeToSend(self, linkedSat):
        """
        Calculates the propagation time of a block going from satellite to satellite.
        """
        distance = linkedSat[0]
        pTime = distance/Vc
        return pTime

    def findNeighbours(self, earth):
        self.linked = None                                                      # Closest sat linked
        self.upper  = earth.LEO[self.in_plane].sats[self.i_in_plane-1]          # Previous sat in the same plane
        if self.i_in_plane < self.n_sat-1:
            self.lower = earth.LEO[self.in_plane].sats[self.i_in_plane+1]       # Following sat in the same plane
        else:
            self.lower = earth.LEO[self.in_plane].sats[0]                       # last satellite of the plane

    def rotate(self, delta_t, longitude, period):
        """
        Rotates the satellite by re-calculating the sperical coordinates, Cartesian coordinates, and longitude and
        latitude adjusted for the new longitude of the orbit, and fraction the elapsed time makes up of the orbit time
        of the satellite.
        """
        # Updating spherical coordinates upon rotation (these are phi, theta before inclination)
        self.phi = longitude
        self.theta = self.theta + 2*math.pi*delta_t/period
        self.theta = self.theta % (2*math.pi)
        
        # Calculating x,y,z coordinates with inclination
        self.x = self.r * (math.sin(self.theta)*math.cos(self.phi) - math.cos(self.theta)*math.sin(self.phi)*math.sin(self.inclination))
        self.y = self.r * (math.sin(self.theta)*math.sin(self.phi) + math.cos(self.theta)*math.cos(self.phi)*math.sin(self.inclination))
        self.z = self.r * math.cos(self.theta)*math.cos(self.inclination)
        self.polar_angle = self.theta  # Angle within orbital plane [radians]
        # updating latitude and longitude after rotation [degrees]
        self.latitude = math.asin(self.z/self.r)  # latitude corresponding to the satellite
        # longitude corresponding to satellite
        if self.x > 0:
            self.longitude = math.atan(self.y/self.x)
        elif self.x < 0 and self.y >= 0:
            self.longitude = math.pi + math.atan(self.y/self.x)
        elif self.x < 0 and self.y < 0:
            self.longitude = math.atan(self.y/self.x) - math.pi
        elif self.y > 0:
            self.longitude = math.pi/2
        elif self.y < 0:
            self.longitude = -math.pi/2
        else:
            self.longitude = 0

    def getLinkLatencies(self, graph):

        latencies = []

        for buffer in self.sendBufferSatsIntra:
            dataRate = nx.path_weight(graph, [self.ID, buffer[2]], "dataRateOG")
            distance = nx.path_weight(graph, [self.ID, buffer[2]], "slant_range")
            bufferSize = len(buffer[1]) + 1

            propTime = distance/Vc
            transmitTime = (bufferSize * 64800) / dataRate

            latency = propTime + transmitTime
            latencies.append([buffer[2], latency])
        for buffer in self.sendBufferSatsInter:
            dataRate = nx.path_weight(graph, [self.ID, buffer[2]], "dataRateOG")
            distance = nx.path_weight(graph, [self.ID, buffer[2]], "slant_range")
            bufferSize = len(buffer[1]) + 1

            propTime = distance / Vc
            transmitTime = (bufferSize * 64800) / dataRate

            latency = propTime + transmitTime
            latencies.append([buffer[2], latency])

        return latencies


'''
    用于建模卫星网络中的“边”(即两个卫星节点之间的连接关系)
    封装了卫星间连接的关键参数(如距离、传输速率等)
    为卫星网络的拓扑构建(如生成图结构)和路由算法(如最短路径计算)提供基础数据支持
'''
class edge:
    def  __init__(self, sati, satj, slant_range, dji, dij, shannonRate):
        self.i = sati   # sati ID
        self.j = satj   # satj ID
        self.slant_range = slant_range  # distance between both sats
        self.dji = dji  # direction from sati to satj
        self.dij = dij  # direction from sati to satj
        self.shannonRate = shannonRate  # max dataRate between sat1 and satj

    def  __repr__(self):
        return '\n node i: {}, node j: {}, slant_range: {}, shannonRate: {}'.format( 
    self.i, 
    self.j,
    self.slant_range,
    self.shannonRate)

    def __cmp__(self, other):
        if hasattr(other, 'slant_range'):    # returns true if has 'weight' attribute
            return self.slant_range.__cmp__(other.slant_range)


'''
    用于建模卫星网络中从网关Ground Terminal, GT 发出的数据块
    通过记录块的传输时间、路径和延迟,为卫星网络的性能分析(如延迟、吞吐量)提供基础数据支持
'''
class DataBlock:
    """
    Class for outgoing block of data from the gateways.
    Instead of simulating the individual data packets from each user, data is gathered at the GTs in blocks - one for
    each destination GT. Once a block is filled with data it is sent as one unit to the destination GT.
    """

    def __init__(self, source, destination, ID, creationTime):
        self.size = 64800  # size in bits
        self.destination = destination
        self.source = source
        self.ID = ID            # a string which holds the source id, destination id, and index of the block, e.g. "1_2_12"
        self.timeAtFull = None  # the simulation time at which the block was full and was ready to be sent.
        self.creationTime = creationTime  # the simulation time at which the block was created.
        self.timeAtFirstTransmission = None  # the simulation time at which the block left the GT.
        self.checkPoints = []   # list of simulation reception times at node with the first entry being the reception time at first sat - can be expanded to include the sat IDs at each checkpoint
        self.checkPointsSend = []   # list of times after the block was sent at each node
        self.path = []
        self.queueLatency = (None, None) # total time acumulated in the queues
        self.txLatency = 0      # total transmission time
        self.propLatency = 0    # total propagation latency
        self.totLatency = 0     # total latency
        self.isNewPath = False
        self.oldPath = []
        self.newPath = []

    def getQueueTime(self):
        '''
        The queue latency is computed in two steps:
        First one: time when the block is sent for the first time - time when the the block is created
        Rest of the steps: sum(checkpoint (Arrival time at node) - checkpointsSend (send time at previous node))
        '''
        queueLatency = [0, []]
        queueLatency[0] += self.timeAtFirstTransmission - self.creationTime        # ANCHOR first step
        queueLatency[1].append(self.timeAtFirstTransmission - self.creationTime)
        for arrived, sendReady in zip(self.checkPoints, self.checkPointsSend):  # rest of the steps
            queueLatency[0] += sendReady - arrived
            queueLatency[1].append(sendReady - arrived)

        self.queueLatency = queueLatency
        return queueLatency

    def getTotalTransmissionTime(self):
        totalTime = 0
        if len(self.checkPoints) == 1:
            return self.checkPoints[0] - self.timeAtFirstTransmission

        lastTime = self.creationTime
        for time in self.checkPoints:
            totalTime += time - lastTime
            lastTime = time
        # ANCHOR KPI: totLatency
        self.totLatency = totalTime 
        return totalTime

    def __repr__(self):
        return'ID = {}\n Source:\n {}\n Destination:\n {}\nTotal latency: {}'.format(
            self.ID,
            self.source,
            self.destination,
            self.totLatency
        )


'''
    建模卫星网络中的地面站,是卫星网络与地面用户的接口
        管理地面站与卫星的连接
        生成并发送数据块 DataBlock 到目标地面站
        接收来自卫星的数据块并记录传输性能
        管理覆盖范围内的地面小区 cell ,确保用户流量的正确路由
'''
class Gateway:
    """
    Class for the gateways (or concentrators). Each gateway will exist as an instance of this class
    which means that each ground station will have separate processes filling and sending blocks to all other GTs.
    """
    def __init__(self, name: str, ID: int, latitude: float, longitude: float, totalX: int, totalY: int, totalGTs, env, totalLocations, earth, pathMetric):
        self.earth = earth

        self.name = name
        self.ID = ID
        self.latitude = latitude  # number is already in degrees
        self.longitude = longitude  # number is already in degrees

        # using the formulas from the set_window() function in the Earth class to the location in terms of cell grid.
        self.gridLocationX = int((0.5 + longitude / 360) * totalX)
        self.gridLocationY = int((0.5 - latitude / 180) * totalY)
        self.cellsInRange = []  # format: [ [(lat,long), userCount, distance], [..], .. ]
        self.totalGTs = totalGTs  # number of GTs including itself
        self.totalLocations = totalLocations # number of possible GTs
        self.totalAvgFlow = None  # total combined average flow from all users in bits per second
        self.totalX = totalX
        self.totalY = totalY

        # cartesian coordinates
        self.polar_angle = (math.pi / 2 - math.radians(self.latitude) + 2 * math.pi) % (2 * math.pi)  # Polar angle in radians
        self.x = Re * math.cos(math.radians(self.longitude)) * math.sin(self.polar_angle)
        self.y = Re * math.sin(math.radians(self.longitude)) * math.sin(self.polar_angle)
        self.z = Re * math.cos(self.polar_angle)

        # satellite linking structure
        self.satsOrdered = []
        self.satIndex = 0
        self.linkedSat = (None, None)  # (distance, sat)
        self.graph = nx.Graph()

        # simpy attributes
        self.env = env  # simulation environment
        self.datBlocks = []  # list of outgoing data blocks - one for each destination GT
        self.fillBlocks = []  # list of simpy processes which fills up the data blocks
        self.sendBlocks = env.process(self.sendBlock())  # simpy process which sends the data blocks
        self.sendBuffer = ([env.event()], [])  # queue of blocks that are ready to be sent
        self.paths = {}  # dictionary for destination: path pairs
        self.pathMetric = pathMetric

        self.receiveFraction = 0

        # comm attributes
        self.dataRate = None
        self.gs2ngeo = RFlink(
            frequency=30e9,
            bandwidth=500e6,
            maxPtx=20,
            aDiameterTx=0.33,
            aDiameterRx=0.26,
            pointingLoss=0.3,
            noiseFigure=2,
            noiseTemperature=290,
            min_rate=10e3
        )

    def makeFillBlockProcesses(self, Receivers, method, fractionIndex):
        """
        Creates the processes for filling the data blocks and adding them to the send-buffer. A separate process for
        each destination gateway is created.
        为每个目标地面站创建一个 fillBlock 进程,用于生成并填充数据块
        """

        self.totalGTs = len(Receivers)
        for receiverIndex, Receiver in enumerate(Receivers):
            if Receiver != self:
                # add a process for each destination which runs the function 'fillBlock'
                self.fillBlocks.append(self.env.process(self.fillBlock(Receiver, method, [fractionIndex, receiverIndex])))

    def fillBlock(self, destination, method, fractionIndex):
        """
        Simpy process function:

        Creates a block headed for a given destination, finds the time for a block to be full and adds the block to the
        send-buffer after the calculated time.

        A separate process for each destination gateway will be running this function.
        负责创建数据块、计算填充时间 timeToFull ,并将填充完成的块添加到发送缓冲区
        """
        index = 0
        unavailableDestinationBuffer = []

        while True:
            try:
                # create a new block to be filled
                block = DataBlock(self, destination, str(self.ID) + "_" + str(destination.ID) + "_" + str(index), self.env.now)

                timeToFull, _ = self.timeToFullBlock(block, method, fractionIndex)

                yield self.env.timeout(timeToFull)  # wait until block is full

                if block.destination.linkedSat[0] is None:
                    unavailableDestinationBuffer.append(block)
                else:
                    while unavailableDestinationBuffer: # empty buffer before adding new block
                        if not self.sendBuffer[0][0].triggered:
                            self.sendBuffer[0][0].succeed()
                            self.sendBuffer[1].append(unavailableDestinationBuffer[0])
                            unavailableDestinationBuffer.pop(0)
                        else:
                            newEvent = self.env.event().succeed()
                            self.sendBuffer[0].append(newEvent)
                            self.sendBuffer[1].append(unavailableDestinationBuffer[0])
                            unavailableDestinationBuffer.pop(0)

                    block.path = self.paths[destination.name]
                    if not block.path:
                        print(self.name, destination.name)
                        exit()
                    block.timeAtFull = self.env.now
                    createdBlocks.append(block)
                    # add block to send-buffer
                    if not self.sendBuffer[0][0].triggered:
                        self.sendBuffer[0][0].succeed()
                        self.sendBuffer[1].append(block)
                    else:
                        newEvent = self.env.event().succeed()
                        self.sendBuffer[0].append(newEvent)
                        self.sendBuffer[1].append(block)
                    index += 1
            except simpy.Interrupt:
                break

    def sendBlock(self):
        """
        Simpy process function:

        Sends data blocks that are filled and added to the send-buffer which is a list of events and data blocks. The
        function monitors the send-buffer, and when the buffer contains one or more triggered events, the function will
        calculate the time it will take to send the block (yet to be implemented), and trigger an event which notifies
        a separate process that a block has been sent (yet to be implemented).

        After a block is sent, the function will send the next, if any more blocks are ready to be sent.

        (While it is assumed that if a buffer is full and ready to be sent it will always be at the first index,
        the method simpy.AnyOf is used. The end result is the same and this method is simple to implement.
        Furthermore, it allows for handling of such errors where a later index is ready but the first is not.
        this case is, however, not handled.)

        Since there is only one link on the GT for sending, there will only be one process running this method.

        从发送缓冲区取出数据块,通过卫星发送到目标
        """
        blockSize = 64800
        while True:
            yield self.sendBuffer[0][0]     # event 0 of block 0

            # wait until a satellite is linked
            while self.linkedSat[0] is None:
                yield self.env.timeout(0.1)

            # calculate propagation time and transmission time
            propTime = self.timeToSend(self.linkedSat)
            timeToSend = blockSize/self.dataRate

            self.sendBuffer[1][0].timeAtFirstTransmission = self.env.now
            yield self.env.timeout(timeToSend)
            # ANCHOR KPI: txLatency send block from GT
            self.sendBuffer[1][0].txLatency += timeToSend

            if not self.sendBuffer[1][0].path:
                print(self.sendBuffer[1][0].source.name, self.sendBuffer[1][0].destination.name)
                exit()

            # if the path metric is "latency", then the graph should be updated to account for current queue sizes
            # at all satellites.
            if self.pathMetric == "latency":
                self.updateGraph(self.sendBuffer[1][0])

            self.linkedSat[1].createReceiveBlockProcess(self.sendBuffer[1][0], propTime)

            # remove from own sendBuffer
            if len(self.sendBuffer[0]) == 1:
                self.sendBuffer[0].pop(0)
                self.sendBuffer[1].pop(0)
                self.sendBuffer[0].append(self.env.event())
            else:
                self.sendBuffer[0].pop(0)
                self.sendBuffer[1].pop(0)

    def timeToSend(self, linkedSat):
        """
            模拟计算发送所用时间
        """
        distance = linkedSat[0]
        pTime = distance/Vc
        return pTime

    def createReceiveBlockProcess(self, block, propTime):
        """
        Function which starts a receiveBlock process upon receiving a block from a transmitter.
        Adds the propagation time to the block attribute
        创建接受数据块的进程
        """

        process = self.env.process(self.receiveBlock(block, propTime))

    def receiveBlock(self, block, propTime):
        """
        Simpy process function:

        This function is used to handle the propagation delay of data blocks. This is done simply by waiting the time
        of the propagation delay. As a GT will always be the last step in a block's path, there is no need to send the
        block further. After the propagation delay, the block is simply added to a list of finished blocks so the KPIs
        can be tracked at the end of the simulation.

        While the transmission delay is handled at the transmitter, the transmitter cannot also wait for the propagation
        delay, otherwise the send-buffer might be overfilled.
        接受数据块进程的实现
        """
        # wait for block to fully propagate
        yield self.env.timeout(propTime)
        # ANCHOR KPI: propLatency send block from GT
        block.propLatency += propTime

        block.checkPoints.append(self.env.now)

        receivedDataBlocks.append(block)

    def cellDistance(self, cell) -> float:
        """
        Calculates the distance to the specified cell (assumed the center of the cell).
        Calculation is based on the geopy package which uses the 'WGS-84' model for earth shape.
        计算当前地面站到特定cell 可以理解成小区 的中心的距离
        """
        cellCoord = (math.degrees(cell.latitude), math.degrees(cell.longitude))  # cell lat and long is saved in a format which is not degrees
        gTCoord = (self.latitude, self.longitude)

        return geopy.distance.geodesic(cellCoord,gTCoord).km

    def distance_GSL(self, satellite):
        """
        Distance between GT and satellite is calculated using the distance formula based on the cartesian coordinates
        in 3D space.
        计算地面站和卫星之间的距离
        """

        satCoords = [satellite.x, satellite.y, satellite.z]
        GTCoords = [self.x, self.y, self.z]

        distance = math.dist(satCoords, GTCoords)
        return distance

    def adjustDataRate(self):
        """
            用于 动态调整地面站 Gateway 与卫星通信数据速率 的核心方法
            基于当前卫星距离、射频链路参数(如发射功率、天线增益)计算信噪比(SNR),并结合预定义的频谱效率阈值,确定实际可用的数据传输速率,确保通信性能符合物理层约束
            不需要修改,是用来模拟真实卫星物理环境的
        """

        speff_thresholds = np.array(
            [0, 0.434841, 0.490243, 0.567805, 0.656448, 0.789412, 0.889135, 0.988858, 1.088581, 1.188304, 1.322253,
             1.487473, 1.587196, 1.647211, 1.713601, 1.779991, 1.972253, 2.10485, 2.193247, 2.370043, 2.458441,
             2.524739, 2.635236, 2.637201, 2.745734, 2.856231, 2.966728, 3.077225, 3.165623, 3.289502, 3.300184,
             3.510192, 3.620536, 3.703295, 3.841226, 3.951571, 4.206428, 4.338659, 4.603122, 4.735354, 4.933701,
             5.06569, 5.241514, 5.417338, 5.593162, 5.768987, 5.900855])
        lin_thresholds = np.array(
            [1e-10, 0.5188000389, 0.5821032178, 0.6266138647, 0.751622894, 0.9332543008, 1.051961874, 1.258925412,
             1.396368361, 1.671090614, 2.041737945, 2.529297996, 2.937649652, 2.971666032, 3.25836701, 3.548133892,
             3.953666201, 4.518559444, 4.83058802, 5.508076964, 6.45654229, 6.886522963, 6.966265141, 7.888601176,
             8.452788452, 9.354056741, 10.49542429, 11.61448614, 12.67651866, 12.88249552, 14.48771854, 14.96235656,
             16.48162392, 18.74994508, 20.18366364, 23.1206479, 25.00345362, 30.26913428, 35.2370871, 38.63669771,
             45.18559444, 49.88844875, 52.96634439, 64.5654229, 72.27698036, 76.55966069, 90.57326009])
        db_thresholds = np.array(
            [-100.00000, -2.85000, -2.35000, -2.03000, -1.24000, -0.30000, 0.22000, 1.00000, 1.45000, 2.23000, 3.10000,
             4.03000, 4.68000, 4.73000, 5.13000, 5.50000, 5.97000, 6.55000, 6.84000, 7.41000, 8.10000, 8.38000, 8.43000,
             8.97000, 9.27000, 9.71000, 10.21000, 10.65000, 11.03000, 11.10000, 11.61000, 11.75000, 12.17000, 12.73000,
             13.05000, 13.64000, 13.98000, 14.81000, 15.47000, 15.87000, 16.55000, 16.98000, 17.24000, 18.10000,
             18.59000, 18.84000, 19.57000])

        pathLoss = 10*np.log10((4*math.pi*self.linkedSat[0]*self.gs2ngeo.f/Vc)**2)
        snr = 10**((self.gs2ngeo.maxPtx_db + self.gs2ngeo.G - pathLoss - self.gs2ngeo.No)/10)
        shannonRate = self.gs2ngeo.B*np.log2(1+snr)

        feasible_speffs = speff_thresholds[np.nonzero(lin_thresholds <= snr)]
        speff = self.gs2ngeo.B*feasible_speffs[-1]

        self.dataRate = speff

    def orderSatsByDist(self, constellation):
        """
        Calculates the distance from the GT to all satellites and saves a sorted (least to greatest distance) list of
        all the satellites that are within range of the GT.
        """
        sats = []
        index = 0
        for orbitalPlane in constellation:
            for sat in orbitalPlane.sats:
                d_GSL = self.distance_GSL(sat)
                # ensure that the satellite is within range
                if d_GSL <= sat.maxSlantRange():
                    sats.append((d_GSL, sat, [index]))
                index += 1
        sats.sort()
        self.satsOrdered = sats

    def addRefOnSat(self):
        """
        Adds a reference of the GT on a satellite based on the local list of satellites that are within range of the GT.
        This function is used in the greedy version of the 'linkSats2GTs()' method in the Earth class.
        The function uses a local indexing number to choose which satellite to add a reference to. If the satellite
        already has a reference, the GT checks if it is closer than the existing reference. If it is closer, it
        overwrites the reference and forces the other GT to add a reference to the next satellite it its own list.
        基于距离排序后的卫星列表,为为地面站分配最近的可用卫星
        若卫星已被其他地面站占用且距离更远,则强制原地面站重新选择下一个卫星
        """
        if self.satIndex >= len(self.satsOrdered):
            self.linkedSat = (None, None)
            print("No satellite for GT {}".format(self.name))
            return

        # check if satellite has reference
        if self.satsOrdered[self.satIndex][1].linkedGT is None:
            # add self as reference on satellite
            self.satsOrdered[self.satIndex][1].linkedGT = self
            self.satsOrdered[self.satIndex][1].GTDist = self.satsOrdered[self.satIndex][0]

        # check if satellites reference is further away than this GT
        elif self.satsOrdered[self.satIndex][1].GTDist < self.satsOrdered[self.satIndex][0]:
            # force other GT to increment satIndex and check next satellite in its local ordered list
            self.satsOrdered[self.satIndex][1].linkedGT.satIndex += 1
            self.satsOrdered[self.satIndex][1].linkedGT.addRefOnSat()

            # add self as reference on satellite
            self.satsOrdered[self.satIndex][1].linkedGT = self
            self.satsOrdered[self.satIndex][1].GTDist = self.satsOrdered[self.satIndex][0]
        else:
            self.satIndex += 1
            if self.satIndex == len(self.satsOrdered):
                self.linkedSat = (None, None)
                print("No satellite for GT {}".format(self.name))
                return

            self.addRefOnSat()

    def link2Sat(self, dist, sat):
        """
        Links the GT to the satellite chosen in the 'linkSats2GTs()' method in the Earth class and makes sure that the
        data rate for the RFlink to the satellite is updated.
        建立地面站与卫星的连接,并调用adjustDataRate,来更新通信速率
        """
        self.linkedSat = (dist, sat)
        sat.linkedGT = self
        sat.GTDist = dist
        self.adjustDataRate()

    def addCell(self, cellInfo):
        """
        Links a cell to the GT by adding the relevant information of the cell to the local list "cellsInRange".
        通过在地面站的本地列表cellsInRange中添加cell的相关信息,来将cell和GT相连
        """
        self.cellsInRange.append(cellInfo)

    def removeCell(self, cell):
        """
        Unused function
        """
        for i, cellInfo in enumerate(self.cellsInRange):
            if cell.latitude == cellInfo[0][0] and cell.longitude == cellInfo[0][1]:
                cellInfo.pop(i)
                return True
        return False

    def findCellsWithinRange(self, earth, maxDistance):
        """
        This function finds the cells that are within the coverage area of the gateway instance. The cells are
        found by checking cells one at a time from the location of the gateway moving outward in a circle until
        the edge of the circle around the terminal exclusively consists of cells that border cells which are outside the
        coverage area. This is an optimized way of finding the cells within the coverage area, as only a limited number
        of cells outside the coverage is checked.

        The size of the area that is checked for is based on the parameter 'maxDistance' which can be seen as the radius
        of the coverage area in kilometers.

        The function will not "link" the cells and the gateway. Instead, it will only add a reference in the
        cells to the closest GT. As a result, all GTs must run this function before any linking is performed. The
        linking is done in the function: "linkCells2GTs()", in the Earth class, which also runs this function. This is
        done to handle cases where the coverage areas of two or more GTs are overlapping and the cells must only link to
        one of the GTs.

        The information added to the "cellsWithinRange" list is used for generating flows from the cells to each GT.
        查找地面站范围内的cell,并保证每个cell仅链接一个地面站
        """

        # Up right:
        isWithinRangeX = True
        x = self.gridLocationX
        while isWithinRangeX:
            y = self.gridLocationY
            isWithinRangeY = True
            if x == earth.total_x: # "roll over" to opposite side of grid.
                x = 0
            cell = earth.cells[x][y]
            distance = self.cellDistance(cell)
            if distance > maxDistance:
                isWithinRangeY = False
                isWithinRangeX = False
            while isWithinRangeY:
                if y == -1:  # "roll over" to opposite side of grid.
                    y = earth.total_y - 1
                cell = earth.cells[x][y]
                distance = self.cellDistance(cell)
                if distance > maxDistance:
                    isWithinRangeY = False
                else:
                    # check if any GT has been added to cell, and if any has check if current GT is closer.
                    if cell.gateway is None or cell.gateway is not None and distance < cell.gateway[1]:
                        # No GT is added to cell or current GT is closer - add current GT.
                        cell.gateway = (self, distance)
                y -= 1  # the y-axis is flipped in the cell grid.
            x += 1

        # Down right:
        isWithinRangeX = True
        x = self.gridLocationX
        while isWithinRangeX:
            y = self.gridLocationY + 1
            isWithinRangeY = True
            if x == earth.total_x:  # "roll over" to opposite side of grid.
                x = 0
            cell = earth.cells[x][y]
            distance = self.cellDistance(cell)
            if distance > maxDistance:
                isWithinRangeY = False
                isWithinRangeX = False
            while isWithinRangeY:
                if y == earth.total_y:  # "roll over" to opposite side of grid.
                    y = 0
                cell = earth.cells[x][y]
                distance = self.cellDistance(cell)
                if distance > maxDistance:
                    isWithinRangeY = False
                else:
                    # check if any GT has been added to cell, and if any has check if current GT is closer.
                    if cell.gateway is None or cell.gateway is not None and distance < cell.gateway[1]:
                        # No GT is added to cell or current GT is closer - add current GT.
                        cell.gateway = (self, distance)
                y += 1  # the y-axis is flipped in the cell grid.
            x += 1

        # up left:
        isWithinRangeX = True
        x = self.gridLocationX - 1
        while isWithinRangeX:
            y = self.gridLocationY
            isWithinRangeY = True
            if x == -1:  # "roll over" to opposite side of grid.
                x = earth.total_x - 1
            cell = earth.cells[x][y]
            distance = self.cellDistance(cell)
            if distance > maxDistance:
                isWithinRangeY = False
                isWithinRangeX = False
            while isWithinRangeY:
                if y == -1:  # "roll over" to opposite side of grid.
                    y = earth.total_y - 1
                cell = earth.cells[x][y]
                distance = self.cellDistance(cell)
                if distance > maxDistance:
                    isWithinRangeY = False
                else:
                    # check if any GT has been added to cell, and if any has check if current GT is closer.
                    if cell.gateway is None or cell.gateway is not None and distance < cell.gateway[1]:
                        # No GT is added to cell or current GT is closer - add current GT.
                        cell.gateway = (self, distance)
                y -= 1  # the y-axis is flipped in the cell grid.
            x -= 1

        # down left:
        isWithinRangeX = True
        x = self.gridLocationX - 1
        while isWithinRangeX:
            y = self.gridLocationY + 1
            isWithinRangeY = True
            if x == -1:  # "roll over" to opposite side of grid.
                x = earth.total_x - 1
            cell = earth.cells[x][y]
            distance = self.cellDistance(cell)
            if distance > maxDistance:
                isWithinRangeY = False
                isWithinRangeX = False
            while isWithinRangeY:
                if y == -1:  # "roll over" to opposite side of grid.
                    y = earth.total_y - 1
                cell = earth.cells[x][y]
                distance = self.cellDistance(cell)
                if distance > maxDistance:
                    isWithinRangeY = False
                else:
                    # check if any GT has been added to cell, and if any has check if current GT is closer.
                    if cell.gateway is None or cell.gateway is not None and distance < cell.gateway[1]:
                        # No GT is added to cell or current GT is closer - add current GT.
                        cell.gateway = (self, distance)
                y += 1  # the y-axis is flipped in the cell grid.
            x -= 1

    def timeToFullBlock(self, block, method, fractionIndex):
        """
        Calculates the average time it will take to fill up a data block and returns the actual time based on a
        random variable following an exponential distribution.
        The method parameter determines how the fractions of the data generation to each destination gateway are handled
        计算填满数据块所需要的平均时间
        """

        if method == "fraction":
            flow  = self.totalAvgFlow * self.earth.fractions[fractionIndex[0], fractionIndex[1]]    # dynamic fractions

        # the next two methods splits the traffic evenly among the active gateways
        elif method == "CurrentNumb":
            # Utilises all the generated traffic regardless of the number of active gateways
            flow = self.totalAvgFlow / (self.totalGTs - 1)

        elif method == "totalNumb":
            # Keeps the fraction to each gateway the same regardless of number of active gateways
            flow = self.totalAvgFlow / (len(self.totalLocations) - 1)

        else:
            print("incorrect method called")
            exit()

        avgTime = block.size / flow  # the average time to fill the buffer in seconds

        time = np.random.exponential(scale=avgTime) # the actual time to fill the buffer after adjustment by exp dist.

        return time, flow

    def getTotalFlow(self, avgFlowPerUser, distanceFunc, maxDistance, capacity = None, fraction = 1.0):
        """
        This function is used as a precursor for the 'timeToFillBlock' method. Based on one of two distance functions
        this function finds the combined average flow from the combined users within the ground coverage area of the GT.

        Calculates the average combined flow from all cells scaling with distance in one of two ways:
            For the step function this means that it essentially just counts the number of users from the local list and
            multiplies with the flowPerUser value.

            For the slope it means that the slope is found using the flowPerUser and maxDistance as the gradient where
            the function gives 0 at the maximum distance.

            If this logic should be changed, it is important that it is done so in accordance with the
            "findCellsWithinRange" method.
        用于计算 地面站 覆盖范围内所有小区的总平均流量 的核心方法,并且考虑了地面站的容量限制
        核心作用是为后续 timeToFillBlock 方法提供流量数据,用于确定数据块填满所需时间
        """

        totalAvgFlow = 0
        avgFlowPerUser = 8593 * 8 # average traffic usage per second in bits

        if distanceFunc == "Step":
            for cell in self.cellsInRange:
                totalAvgFlow += cell[1] * avgFlowPerUser

        elif distanceFunc == "Slope":
            gradient = (0-avgFlowPerUser)/(maxDistance-0)
            for cell in self.cellsInRange:
                totalAvgFlow += (gradient * cell[2] + avgFlowPerUser) * cell[1]

        else:
            print("Error, distance function not recognized. Provided function = {}. Allowed functions: {} or {}".format(
                distanceFunc,
                "Step",
                "slope"))
            exit()

        if self.linkedSat[0] is None:
            self.dataRate = self.gs2ngeo.min_rate

        if not capacity:
            capacity = self.dataRate

        if totalAvgFlow < capacity * fraction:
            self.totalAvgFlow = totalAvgFlow
        else:
            self.totalAvgFlow = capacity * fraction

    def updateGraph(self, block):
        """
        This function is used when a block is about to be transmitted if the path metric is set to "latency".
        The function updates the weights of the constellation graph to take into account the current queue sizes at all
        satellites. It also finds the new shortest path for the block based on the new weights.
        动态更新星座图 Constellation Graph 边权重,并重新计算数据块最短路径 的核心方法
        当路径选择策略基于“延迟 latency ”时,根据卫星当前队列状态调整图中边的延迟权重,确保路径选择反映网络实时负载,优化数据块传输延迟
        """
        for plane in self.earth.LEO:
            for satellite in plane.sats:
                latencies = satellite.getLinkLatencies(self.graph)
                for latency in latencies:
                    nx.set_edge_attributes(self.graph, {(satellite.ID, latency[0]): {'latency': latency[1]}})
        block.path = getShortestPath(self.name, block.destination.name, "latency", self.graph)

    def __eq__(self, other):
        if self.latitude == other.latitude and self.longitude == other.longitude:
            return True
        else:
            return False

    def __repr__(self):
        return 'Location = {}\n Longitude = {}\n Latitude = {}\n pos x= {}, pos y= {}, pos z= {}'.format(
            self.name,
            self.longitude,
            self.latitude,
            self.x,
            self.y,
            self.z)


# A single cell on earth
'''
    用于建模卫星通信覆盖区域中的 地面小区 ,每个小区代表地球表面的一个地理区域,包含用户分布、地理位置、通信参数等信息
'''
class Cell:
    def __init__(self, total_x, total_y, cell_x, cell_y, users, Re=6378e3, f=20e9, bw=200e6, noise_power=1 / (1e11)):
        # X and Y coordinates of the cell on the dataset map
        self.map_x = cell_x
        self.map_y = cell_y
        # Latitude and longitude of the cell as per dataset map
        self.latitude = math.pi * (0.5 - cell_y / total_y)
        self.longitude = (cell_x / total_x - 0.5) * 2 * math.pi
        if self.latitude < -5 or self.longitude < -5:
            print("less than 0")
            print(self.longitude, self.latitude)
            print(cell_x, cell_y)
        # Actual area the cell covers on earth (scaled for)
        self.area = 4 * math.pi * Re * Re * math.cos(self.latitude) / (total_x * total_y)
        # X,Y,Z coordinates to the center of the cell (assumed)
        self.x = Re * math.cos(self.latitude) * math.cos(self.longitude)
        self.y = Re * math.cos(self.latitude) * math.sin(self.longitude)
        self.z = Re * math.sin(self.latitude)

        self.users = users  # Population in the cell
        self.f = f  # Frequency used by the cell
        self.bw = bw  # Bandwidth used for the cell
        self.noise_power = noise_power  # Noise power for the cell
        self.rejected = True  # Usefulfor applications process to show if the cell is rejected or accepted
        self.gateway = None  # (groundstation, distance)

    def __repr__(self):
        '''
            返回小区的可读字符串描述,包含用户数、面积(转换为平方公里)、经纬度(度数)、三维坐标及地图坐标
        '''
        return 'Users = {}\n area = {} km^2\n longitude = {} deg\n latitude = {} deg\n pos x = {}\n pos y = {}\n pos ' \
               'z = {}\n x position on map = {}\n y position on map = {}'.format(
                self.users,
                '%.2f' % (self.area / 1e6),
                '%.2f' % math.degrees(self.longitude),
                '%.2f' % math.degrees(self.latitude),
                '%.2f' % self.x,
                '%.2f' % self.y,
                '%.2f' % self.z,
                '%.2f' % self.map_x,
                '%.2f' % self.map_y)

    def setGT(self, gateways, maxDistance = 60):
        """
        Finds the closest gateway and updates the internal attribute 'self.gateway' as a tuple:
        (Gateway, distance to terminal). If the distance to the closest gateway is less than some maximum
        distance, the cell information is added to the gateway.
        找到最近的地面站并更新 self.gateway,若距离≤maxDistance则将小区加入地面站覆盖列表,否则标记用户数为0
        """
        closestGT = (gateways[0], gateways[0].cellDistance(self))
        for gateway in gateways[1:]:
            distanceToGT = gateway.cellDistance(self)
            if distanceToGT < closestGT[1]:
                closestGT = (gateway, distanceToGT)
        self.gateway = closestGT

        if closestGT[1] <= maxDistance:
            closestGT[0].addCell([(math.degrees(self.latitude), math.degrees(self.longitude)), self.users, closestGT[1]])
        else:
            self.users = 0
        return closestGT


# Earth consisting of cells
'''
    建模地球表面的地理环境、用户分布(通过小区 Cell )、地面站 Gateway 部署,以及卫星星座( LEO )的动态管理
        加载人口分布数据(通过TIFF图像),生成地面小区网格
        管理地面站与小区的覆盖关联 linkCells2GTs
        管理卫星与地面站的连接 linkSats2GTs 
        在星座移动时动态更新卫星的传输进程 updateSatelliteProcessesSimpler updateSatelliteProcessesCorrect
'''
class Earth:
    def __init__(self, env, img_path, gt_path, constellation, inputParams, deltaT, totalLocations, getRates = False, window=None):
        '''
            初始化地球对象,加载人口数据生成小区网格,导入地面站数据,初始化数据块生成进程,并创建卫星星座
                通过TIFF图像加载人口分布,生成二维小区网格 self.cells
                根据输入参数 inputParams 筛选或全选地面站 self.gateways 
                启动卫星星座移动的Simpy进程 self.moveConstellation
        '''
        # Input the population count data
        # img_path = 'Population Map/gpw_v4_population_count_rev11_2020_15_min.tif'

        pop_count_data = Image.open(img_path)

        pop_count = np.array(pop_count_data)
        pop_count[pop_count < 0] = 0  # ensure there are no negative values

        # total image sizes
        [self.total_x, self.total_y] = pop_count_data.size

        self.total_cells = self.total_x * self.total_y

        # List of all cells stored in a 2d array as per the order in dataset
        self.cells = []
        for i in range(self.total_x):
            self.cells.append([])
            for j in range(self.total_y):
                self.cells[i].append(Cell(self.total_x, self.total_y, i, j, pop_count[j][i]))

        # window is a list with the coordinate bounds of our window of interest
        # format for window = [western longitude, eastern longitude, southern latitude, northern latitude]
        if window is not None:  # if window provided
            # latitude, longitude bounds:
            self.lati = [window[2], window[3]]
            self.longi = [window[0], window[1]]
            # dataset pixel bounds:
            self.windowx = (
            (int)((0.5 + window[0] / 360) * self.total_x), (int)((0.5 + window[1] / 360) * self.total_x))
            self.windowy = (
            (int)((0.5 - window[3] / 180) * self.total_y), (int)((0.5 - window[2] / 180) * self.total_y))
        else:  # set window size as entire world if no window provided
            self.lati = [-90, 90]
            self.longi = [-179, 180]
            self.windowx = (0, self.total_x)
            self.windowy = (0, self.total_y)

        # import gateways from .csv
        self.gateways = []

        gateways = pd.read_csv(gt_path)

        length = 0
        for i, location in enumerate(gateways['Location']):
            for name in inputParams['Locations']:
                if name in location.split(","):
                    length += 1

        if inputParams['Locations'][0] != 'All':
            for i, location in enumerate(gateways['Location']):
                for name in inputParams['Locations']:
                    if name in location.split(","):
                        lName = gateways['Location'][i]
                        gtLati = gateways['Latitude'][i]
                        gtLongi = gateways['Longitude'][i]
                        self.gateways.append(Gateway(lName, i, gtLati, gtLongi, self.total_x, self.total_y,
                                                     length, env, totalLocations, self, inputParams['Pathing'][0]))
                        break
        else:
            for i in range(len(gateways['Latitude'])):
                name = gateways['Location'][i]
                gtLati = gateways['Latitude'][i]
                gtLongi = gateways['Longitude'][i]
                self.gateways.append(Gateway(name, i, gtLati, gtLongi, self.total_x, self.total_y,
                                             len(gateways['Latitude']), env, totalLocations, self, inputParams['Pathing'][0]))

        # If only select gateways are needed for generating and receiving, uncomment the below code and set the desired
        # fractions of generators and receivers.
        # Furthermore, replace "self.gateways" with self.generators in the definition of the for loop
        # and self.receivers as the parameter in the "gt.makeFillBlockProcess()" method.

        # self.generatorGTs = self.gateways[:int(len(self.gateways) / 2)]
        # self.receiverGTs = self.gateways[int(len(self.gateways) / 2):]

        # initialise processes to create data Blocks on generator GTs.
        if not getRates:
            for gtIndex, gt in enumerate(self.gateways):
                gt.makeFillBlockProcesses(self.gateways, "totalNumb", gtIndex)

        # initialize fractions table - uncomment if generators and receivers are used.
        # self.fractions = np.zeros([len(self.generatorGTs), len(self.receiverGTs)])

        # create constellation of satellites
        self.LEO = create_Constellation(constellation, env)

        self.pathParam = inputParams['Pathing'][0]

        # Simpy process for handling moving the constellation and the satellites within the constellation
        self.moveConstellation = env.process(self.moveConstellation(env, deltaT, getRates))

    def set_window(self, window):  # function to change/set window for the earth
        """
        Unused function
        """
        self.lati = [window[2], window[3]]
        self.longi = [window[0], window[1]]
        self.windowx = ((int)((0.5 + window[0] / 360) * self.total_x), (int)((0.5 + window[1] / 360) * self.total_x))
        self.windowy = ((int)((0.5 - window[3] / 180) * self.total_y), (int)((0.5 - window[2] / 180) * self.total_y))

    def linkCells2GTs(self, distance):
        """
        Finds the cells that are within the coverage areas of all GTs and links them ensuring that a cell only links to
        a single GT.
        找到所有地面站覆盖范围内的小区并关联,确保每个小区仅连接一个地面站
        """
        start = time.time()

        # Find cells that are within range of all GTs
        for i, gt in enumerate(self.gateways):
            print("Finding cells within coverage area of GT {} of {}".format(i+1, len(self.gateways)), end='\r')
            gt.findCellsWithinRange(self, distance)
        print('\r')
        print("Time taken to find cells that are within range of all GTs: {} seconds".format(time.time() - start))

        start = time.time()

        # Add reference for cells to the GT they are closest to
        # 为小区添加最近地面站的引用
        for cells in self.cells:
            for cell in cells:
                if cell.gateway is not None:
                    cell.gateway[0].addCell([(math.degrees(cell.latitude),
                                                     math.degrees(cell.longitude)),
                                                    cell.users,
                                                    cell.gateway[1]])

        print("Time taken to add cell information to all GTs: {} seconds".format(time.time() - start))  # 输出耗时
        print()

    def linkSats2GTs(self, method):
        """
        Links GTs to satellites. One satellite is only allowed to link to one GT.
        将地面站与卫星连接。每个卫星仅允许连接一个地面站
        支持两种策略
            Greedy(贪心) ：每个地面站选择最近的卫星；
            Optimize(优化) ：通过最小权匹配( linear_sum_assignment )全局优化连接。
        """
        sats = []
        for orbit in self.LEO:
            for sat in orbit.sats:
                sat.linkedGT = None
                sat.GTDist = None
                sats.append(sat)

        if method == "Greedy":  # 贪心算法
            for GT in self.gateways:
                GT.orderSatsByDist(self.LEO)
                GT.addRefOnSat()

            for orbit in self.LEO:
                for sat in orbit.sats:
                    if sat.linkedGT is not None:
                        sat.linkedGT.link2Sat(sat.GTDist, sat)
        elif method == "Optimize":  # 优化算法(最小权匹配)
            # make cost matrix
            SxGT = np.array([[99999 for _ in range(len(sats))] for _ in range(len(self.gateways))])
            for i, GT in enumerate(self.gateways):
                GT.orderSatsByDist(self.LEO)
                for val, entry in enumerate(GT.satsOrdered):
                    SxGT[i][entry[2][0]] = val

            # find assignment of GSL which minimizes the cost from the cost matrix
            # 求解最小权匹配(线性和分配问题)
            rowInd, colInd = linear_sum_assignment(SxGT)

            # link satellites and GTs
            for i, GT in enumerate(self.gateways):
                if SxGT[rowInd[i]][colInd[i]] < len(GT.satsOrdered):
                    sat = GT.satsOrdered[SxGT[rowInd[i]][colInd[i]]]
                    GT.link2Sat(sat[0], sat[1])
                else:
                    GT.linkedSat = (None, None)

    def getCellUsers(self):
        """
            Used for plotting the population map.
            返回所有小区的用户数,用于绘制人口分布地图
        """
        temp = []
        for i, cellList in enumerate(self.cells):
            temp.append([])
            for cell in cellList:
                temp[i].append(cell.users)
        return temp

    def updateSatelliteProcessesSimpler(self, graph):
        """
        Can be used for a simpler version of updating the processes on satellites. However, it does not take into
        account that some processes may be able to continue without being stopped. Stopping the processes may lose
        time of the transmission of a block.

        Function which ensures all processes on all satellites are updated after constellation movement. This is done in
        several steps:
            - All blocks waiting to be sent or currently being sent has their paths updated.
            - All processes are stopped and remade according to current links - all transmission progress is lost on
            blocks currently being transmitted.
            - All buffers are emptied and blocks are redistributed to new buffers according to the blocks' arrival time
            at the satellite.
        可用于卫星进程更新的简化版本。但未考虑部分进程可能无需停止即可继续运行。停止进程可能丢失正在传输的块的时间。

        该函数确保星座移动后所有卫星的进程更新,步骤如下：
            - 所有等待发送或正在发送的块更新路径；
            - 停止所有进程并根据当前链路重新创建(正在传输的块的传输进度丢失)；
            - 清空所有缓冲区,根据块到达卫星的时间重新分配到新缓冲区。
        """

        # update ISL references in all satellites, adjust data rate to GTs and ensure send-processes are correct
        sats = []
        for plane in self.LEO:
            for sat1 in plane.sats:
                sats.append(sat1)
        for plane in self.LEO:
            for sat in plane.sats:

                # remake path for all blocks
                for buffer in sat.sendBufferSatsIntra:
                    for block in buffer[1]:
                        destination = block.destination.name
                        newPath = getShortestPath(sat.ID, destination, self.pathParam, graph)
                        path = None
                        # splice old and new path
                        for i, step in enumerate(block.path):
                            if step[0] == sat.ID:
                                path = block.path[:i] + newPath
                                break
                        if path is None:
                            print("no path to sat:")
                            print(block)
                            exit()
                        block.path = path
                for buffer in sat.sendBufferSatsInter:
                    for block in buffer[1]:
                        destination = block.destination.name
                        newPath = getShortestPath(sat.ID, destination, self.pathParam, graph)
                        path = None
                        # splice old and new path
                        for i, step in enumerate(block.path):
                            if step[0] == sat.ID:
                                path = block.path[:i] + newPath
                                break
                        if path is None:
                            print("no path to sat:")
                            print(block)
                            exit()
                        block.path = path
                for block in sat.sendBufferGT[1]:
                    destination = block.destination.name
                    newPath = getShortestPath(sat.ID, destination, self.pathParam, graph)
                    path = None
                    # splice old and new path
                    for i, step in enumerate(block.path):
                        if step[0] == sat.ID:
                            path = block.path[:i] + newPath
                            break
                    if path is None:
                        print("no path to GT:")
                        print(block)
                        exit()
                    block.path = path
                for block in sat.tempBlocks:
                    destination = block.destination.name
                    newPath = getShortestPath(sat.ID, destination, self.pathParam, graph)
                    path = None
                    # splice old and new path
                    for i, step in enumerate(block.path):
                        if step[0] == sat.ID:
                            path = block.path[:i] + newPath
                            break
                    if path is None:
                        print("no path from Temp:")
                        print(block)
                        exit()
                    block.path = path

                # find neighboring satellites
                neighbors = list(nx.neighbors(graph, sat.ID))
                itt = 0
                neighborSats = []
                for sat2 in sats:
                    if sat2.ID in neighbors:
                        dataRate = nx.path_weight(graph, [sat2.ID, sat.ID], "dataRateOG")
                        distance = nx.path_weight(graph, [sat2.ID, sat.ID], "slant_range")
                        neighborSats.append((distance, sat2, dataRate))
                        itt += 1
                        if itt == len(neighbors):
                            break

                sat.intraSats = []
                sat.interSats = []

                # add new satellites as references
                for neighbor in neighborSats:
                    if neighbor[1].in_plane == sat.in_plane:
                        sat.intraSats.append(neighbor)
                    else:
                        sat.interSats.append(neighbor)

                # stop all processes
                for process in sat.sendBlocksSatsInter:
                    process.interrupt()
                for process in sat.sendBlocksSatsIntra:
                    process.interrupt()
                for process in sat.sendBlocksGT:
                    process.interrupt()
                sat.sendBlocksSatsIntra = []
                sat.sendBlocksSatsInter = []
                sat.sendBlocksGT = []

                # add all blocks to list and reset queues
                blocksToDistribute = []
                for buffer in sat.sendBufferSatsIntra:
                    for block in buffer[1]:
                        blocksToDistribute.append((block.checkPoints[-1], block))
                sat.sendBufferSatsIntra = []
                for buffer in sat.sendBufferSatsInter:
                    for block in buffer[1]:
                        blocksToDistribute.append((block.checkPoints[-1], block))
                sat.sendBufferSatsInter = []
                for block in sat.sendBufferGT[1]:
                    blocksToDistribute.append((block.checkPoints[-1], block))
                sat.sendBufferGT = ([sat.env.event()], [])

                # remake all processes
                if sat.linkedGT is not None:
                    sat.adjustDownRate()
                    # make a process for the GSL from sat to GT
                    sat.sendBlocksGT.append(sat.env.process(sat.sendBlock((sat.GTDist, sat.linkedGT), False)))

                for neighbor in sat.intraSats:
                    # make a send buffer for each ISL ([self.env.event()], [DataBlock(0, 0, "0", 0)], 0)
                    sat.sendBufferSatsIntra.append(([sat.env.event()], [], neighbor[1].ID))

                    # make a process for each ISL
                    sat.sendBlocksSatsIntra.append(sat.env.process(sat.sendBlock(neighbor, True, True)))

                for neighbor in sat.interSats:
                    # make a send buffer for each ISL ([self.env.event()], [DataBlock(0, 0, "0", 0)], 0)
                    sat.sendBufferSatsInter.append(([sat.env.event()], [], neighbor[1].ID))

                    # make a process for each ISL
                    sat.sendBlocksSatsInter.append(sat.env.process(sat.sendBlock(neighbor, True, False)))

                # sort blocks by arrival time at satellite
                blocksToDistribute.sort()
                # add blocks to the correct queues based on next step in their path
                # since the blocks list is sorted by arrival time, the order in the new queues is correct
                for block in blocksToDistribute:
                    # get this satellite's index in the blocks path
                    index = None
                    for i, step in enumerate(block[1].path):
                        if sat.ID == step[0]:
                            index = i

                    # check if next step in path is GT (last step in path)
                    if index == len(block[1].path) - 2:
                        # add block to GT send-buffer
                        if not sat.sendBufferGT[0][0].triggered:
                            sat.sendBufferGT[0][0].succeed()
                            sat.sendBufferGT[1].append(block[1])
                        else:
                            newEvent = sat.env.event().succeed()
                            sat.sendBufferGT[0].append(newEvent)
                            sat.sendBufferGT[1].append(block[1])
                    else:
                        # get ID of next sat and find if it is intra or inter
                        ID = None
                        isIntra = False
                        for neighborSat in sat.intraSats:
                            id = neighborSat[1].ID
                            if id == block[1].path[index + 1][0]:
                                ID = neighborSat[1].ID
                                isIntra = True
                        for neighborSat in sat.interSats:
                            id = neighborSat[1].ID
                            if id == block[1].path[index + 1][0]:
                                ID = neighborSat[1].ID

                        if ID is not None:
                            sendBuffer = None
                            # find send-buffer for the satellite
                            if isIntra:
                                for buffer in sat.sendBufferSatsIntra:
                                    if ID == buffer[2]:
                                        sendBuffer = buffer
                            else:
                                for buffer in sat.sendBufferSatsInter:
                                    if ID == buffer[2]:
                                        sendBuffer = buffer

                            # add block to buffer
                            if not sendBuffer[0][0].triggered:
                                sendBuffer[0][0].succeed()
                                sendBuffer[1].append(block[1])
                            else:
                                newEvent = sat.env.event().succeed()
                                sendBuffer[0].append(newEvent)
                                sendBuffer[1].append(block[1])
                        else:
                            print("buffer for next satellite in path could not be found")

    def updateSatelliteProcessesCorrect(self, graph):
        """
        Function which ensures all processes on all satellites are updated after constellation movement. This is done in
        several steps:
            - All blocks waiting to be sent or currently being sent has their paths updated.
            - ISLs are updated with references to new inter-orbit satellites (intra-orbit links do not change).
                - This includes updating buffer if ISL is changed
                - It also includes remaking send-process if ISL is changed
                - Despite intra-orbit links not changing, blocks in an intra-orbit buffer may have to be moved.
            - GSL is updated:
                - Depending on new status - whether the satellite has a GSL or not - and past status - whether the
                satellite had a GSL or not - GSL buffer and process is handled accordingly.
            - All blocks not currently being transmitted to a satellite/GT, which is still present as a ISL or GSL, are
            redistributed to send-buffers according to their arrival time at the satellite.

        This function differentiates from the simple version by allowing continued operation of send-processes after
        constellation movement if the link is not broken.
        该函数用于确保星座移动后所有卫星上的进程得到更新。这一过程通过以下几个步骤完成：
            - 所有等待发送或当前正在发送的数据块的路径将被更新。
            - 星间链路(ISL)将更新为指向新的跨轨道卫星引用(同轨道链路不会改变)。
                - 这包括在ISL变更时更新缓冲区
                - 还包括在ISL变更时重新创建发送进程
                - 尽管同轨道链路不会改变,但同轨道缓冲区中的数据块可能需要移动。
            - 地面站链路(GSL)将被更新：
                - 根据新状态(卫星是否连接GSL)和旧状态(卫星之前是否连接GSL),相应处理GSL缓冲区和进程。
            - 所有未被当前传输到卫星/地面站(且该链路仍作为ISL或GSL存在)的数据块,将根据其到达卫星的时间重新分配到发送缓冲区。

        该函数与简化版本的区别在于：如果链路未断开,允许星座移动后继续运行现有的发送进程。
        """
        sats = []
        for plane in self.LEO:
            for sat1 in plane.sats:
                sats.append(sat1)

        for plane in self.LEO:
            for sat in plane.sats:
                # remake path for all blocks
                for buffer in sat.sendBufferSatsIntra:
                    index = 0
                    while index < len(buffer[1]):
                        block = buffer[1][index]
                        destination = block.destination.name
                        newPath = getShortestPath(sat.ID, destination, self.pathParam, graph)
                        if newPath == -1:
                            if len(buffer[0]) == 1:
                                buffer[0].pop(index)
                                buffer[1].pop(index)
                                buffer[0].append(sat.env.event())
                            else:
                                buffer[0].pop(index)
                                buffer[1].pop(index)
                            continue
                        path = None
                        # splice old and new path
                        for i, step in enumerate(block.path):
                            if step[0] == sat.ID:
                                path = block.path[:i] + newPath
                                break
                        if path is None:
                            print("no path to sat:")
                            print(block)
                            exit()
                        block.isNewPath = True
                        block.oldPath = block.path
                        block.newPath = newPath
                        block.path = path
                        index += 1

                for buffer in sat.sendBufferSatsInter:
                    index = 0
                    while index < len(buffer[1]):
                        block = buffer[1][index]
                        destination = block.destination.name
                        newPath = getShortestPath(sat.ID, destination, self.pathParam, graph)
                        if newPath == -1:
                            if len(buffer[0]) == 1:
                                buffer[0].pop(index)
                                buffer[1].pop(index)
                                buffer[0].append(sat.env.event())
                            else:
                                buffer[0].pop(index)
                                buffer[1].pop(index)
                            continue
                        path = None
                        # splice old and new path
                        for i, step in enumerate(block.path):
                            if step[0] == sat.ID:
                                path = block.path[:i] + newPath
                                break
                        if path is None:
                            print("no path to sat:")
                            print(block)
                            exit()
                        block.isNewPath = True
                        block.oldPath = block.path
                        block.newPath = newPath
                        block.path = path
                        index += 1

                index = 0
                while index < len(sat.sendBufferGT[1]):
                    block = sat.sendBufferGT[1][index]
                    destination = block.destination.name
                    newPath = getShortestPath(sat.ID, destination, self.pathParam, graph)
                    if newPath == -1:
                        if len(sat.sendBufferGT[0]) == 1:
                            sat.sendBufferGT[0].pop(index)
                            sat.sendBufferGT[1].pop(index)
                            sat.sendBufferGT[0].append(sat.env.event())
                        else:
                            sat.sendBufferGT[0].pop(index)
                            sat.sendBufferGT[1].pop(index)
                        continue
                    path = None
                    # splice old and new path
                    for i, step in enumerate(block.path):
                        if step[0] == sat.ID:
                            path = block.path[:i] + newPath
                            break
                    if path is None:
                        print("no path to GT:")
                        print(block)
                        exit()
                    block.isNewPath = True
                    block.oldPath = block.path
                    block.newPath = newPath
                    block.path = path
                    index += 1

                index = 0
                while index < len(sat.tempBlocks):
                    block = sat.tempBlocks[index]
                    destination = block.destination.name
                    newPath = getShortestPath(sat.ID, destination, self.pathParam, graph)

                    if newPath == -1:
                        block.path = -1
                        if len(sat.tempBlocks[0]) == 1:
                            sat.tempBlocks[0].pop(index)
                            sat.tempBlocks[1].pop(index)
                            sat.tempBlocks[0].append(sat.env.event())
                        else:
                            sat.tempBlocks[0].pop(index)
                            sat.tempBlocks[1].pop(index)
                        continue

                    path = None
                    # splice old and new path
                    for i, step in enumerate(block.path):
                        if step[0] == sat.ID:
                            path = block.path[:i] + newPath
                            break
                    if path is None:
                        print("no path from Temp:")
                        print(block)
                        exit()
                    block.isNewPath = True
                    block.oldPath = block.path
                    block.newPath = newPath
                    block.path = path
                    index += 1

                # find neighboring satellites
                neighbors = list(nx.neighbors(graph, sat.ID))
                itt = 0
                neighborSatsInter = []
                for sat2 in sats:
                    if sat2.ID in neighbors:
                        # we only care about the satellite if it is an inter-plane ISL
                        # we assume intra-plane ISLs will not change
                        if sat2.in_plane != sat.in_plane:
                            dataRate = nx.path_weight(graph, [sat2.ID, sat.ID], "dataRateOG")
                            distance = nx.path_weight(graph, [sat2.ID, sat.ID], "slant_range")
                            neighborSatsInter.append((distance, sat2, dataRate))
                        itt += 1
                        if itt == len(neighbors):
                            break
                sat.interSats = neighborSatsInter
                # list of blocks to be redistributed
                blocksToDistribute = []

                ### inter-plane ISLs ###

                sat.newBuffer = [True for _ in range(len(neighborSatsInter))]

                # make a list of False entries for each current neighbor
                sameSats = [False for _ in range(len(neighborSatsInter))]

                buffers = [None for _ in range(len(neighborSatsInter))]
                processes = [None for _ in range(len(neighborSatsInter))]

                # go through each process/buffer
                #   - check if the satellite is still there:
                #       - if it is, change the corresponding False to True, handle blocks and add process and buffer references to temporary list
                #       - if it is not, remove blocks from buffer and stop process
                for bufferIndex, buffer in enumerate(sat.sendBufferSatsInter):
                    # check if the satellite is still there
                    isPresent = False
                    for neighborIndex, neighbor in enumerate(neighborSatsInter):
                        if buffer[2] == neighbor[1].ID:
                            isPresent = True
                            sameSats[neighborIndex] = True

                            ## handle blocks
                            # check if there are blocks in the buffer
                            if buffer[1]:
                                # find index of satellite in block's path
                                index = None
                                for i, step in enumerate(buffer[1][0].path):
                                    if sat.ID == step[0]:
                                        index = i
                                        break

                                # check if next step in path corresponds to buffer's satellite
                                if buffer[1][0].path[index + 1][0] == buffer[2]:
                                    # add all but the first block to redistribution list
                                    for block in buffer[1][1:]:
                                        blocksToDistribute.append((block.checkPoints[-1], block))

                                    # add buffer with only first block present to temp list
                                    buffers[neighborIndex] = ([sat.env.event().succeed()], [sat.sendBufferSatsInter[bufferIndex][1][0]], buffer[2])
                                    processes[neighborIndex] = sat.sendBlocksSatsInter[bufferIndex]
                                else:
                                    # add all blocks to redistribution list
                                    for block in buffer[1]:
                                        blocksToDistribute.append((block.checkPoints[-1], block))
                                    # reset buffer
                                    buffers[neighborIndex] = ([sat.env.event()], [], buffer[2])

                                    # reset process
                                    sat.sendBlocksSatsInter[bufferIndex].interrupt()
                                    processes[neighborIndex] = sat.env.process(sat.sendBlock(neighbor, True, False))

                            else: # there are no blocks in the buffer
                                # add buffer and remake process
                                buffers[neighborIndex] = sat.sendBufferSatsInter[bufferIndex]
                                sat.sendBlocksSatsInter[bufferIndex].interrupt()
                                processes[neighborIndex] = sat.env.process(sat.sendBlock(neighbor, True, False))
                                # sendBlocksSatsInter[bufferIndex]

                            break
                    if not isPresent:
                        # add blocks to redistribution list
                        for block in buffer[1]:
                            blocksToDistribute.append((block.checkPoints[-1], block))
                        # stop process
                        sat.sendBlocksSatsInter[bufferIndex].interrupt()

                # make buffer and process for new neighbors(s)
                # - go through list of previously false entries:
                #   - check  entry for each neighbor:
                #       - if False, create buffer and process for new neighbor
                # - clear temporary list of processes and buffers
                for entryIndex, entry in enumerate(sameSats):
                    if not entry:
                        buffers[entryIndex] = ([sat.env.event()], [], neighborSatsInter[entryIndex][1].ID)
                        processes[entryIndex] = sat.env.process(sat.sendBlock(neighborSatsInter[entryIndex], True, False))

                # overwrite buffers and processes
                sat.sendBlocksSatsInter = processes
                sat.sendBufferSatsInter = buffers

                ### intra-plane ISLs ###
                # check blocks for each buffer
                for bufferIndex, buffer in enumerate(sat.sendBufferSatsIntra):
                    ## handle blocks
                    # check if there are blocks in the buffer
                    if buffer[1]:
                        # find index of satellite in block's path
                        index = None
                        for i, step in enumerate(buffer[1][0].path):
                            if sat.ID == step[0]:
                                index = i
                                break

                        # check if next step in path corresponds to buffer's satellite
                        if buffer[1][0].path[index + 1][0] == buffer[2]:
                            # add all but the first block to redistribution list
                            for block in buffer[1][1:]:
                                blocksToDistribute.append((block.checkPoints[-1], block))

                            # remove all but the first block and event from the buffer
                            length = len(sat.sendBufferSatsIntra[bufferIndex][1]) - 1
                            for _ in range(length):
                                sat.sendBufferSatsIntra[bufferIndex][1].pop(1)
                                sat.sendBufferSatsIntra[bufferIndex][0].pop(1)

                        else:
                            # add all blocks to redistribution list
                            for block in buffer[1]:
                                blocksToDistribute.append((block.checkPoints[-1], block))
                            # reset buffer
                            sat.sendBufferSatsIntra[bufferIndex] = ([sat.env.event()], [], buffer[2])

                            # reset process
                            sat.sendBlocksSatsIntra[bufferIndex].interrupt()
                            sat.sendBlocksSatsIntra[bufferIndex] = sat.env.process(sat.sendBlock(sat.intraSats[bufferIndex], True, True))

                ### GSL ###
                # check if satellite has a linked GT
                if sat.linkedGT is not None:
                    sat.adjustDownRate()

                    # check if it had a sendBlocksGT process
                    if sat.sendBlocksGT:
                        # check if there are any blocks in the buffer
                        if sat.sendBufferGT[1]:
                            # check if linked GT is the same as the destination of first block in sendBufferGT
                            if sat.sendBufferGT[1][0].destination != sat.linkedGT:
                                sat.sendBlocksGT[0].interrupt()
                                sat.sendBlocksGT = []

                                # remove blocks from queue and add to list of blocks which should be redistributed
                                for block in sat.sendBufferGT[1]:
                                    blocksToDistribute.append(
                                        (block.checkPoints[-1], block))  # (latest checkpoint time, block)
                                sat.sendBufferGT = ([sat.env.event()], [])

                                # make new send process for new linked GT
                                sat.sendBlocksGT.append(
                                    sat.env.process(sat.sendBlock((sat.GTDist, sat.linkedGT), False)))
                            else:
                                # keep the first block in the buffer and let process continue
                                for block in sat.sendBufferGT[1][1:]:
                                    blocksToDistribute.append(
                                        (block.checkPoints[-1], block))  # (latest checkpoint time, block)
                                length = len(sat.sendBufferGT[1]) - 1
                                for _ in range(length):
                                    sat.sendBufferGT[1].pop(1) # pop all but the first block
                                    sat.sendBufferGT[0].pop(1) # pop all but the first event

                        else:  # there are no blocks in the buffer
                            sat.sendBlocksGT[0].interrupt()
                            sat.sendBlocksGT = []
                            sat.sendBufferGT = ([sat.env.event()], [])
                            # make new send process for new linked GT
                            sat.sendBlocksGT.append(sat.env.process(sat.sendBlock((sat.GTDist, sat.linkedGT), False)))

                    else:  # it had no process running
                        # there should be no blocks in the GT buffer, but just in case - if there are none, then the for loop will not run
                        # remove blocks from queue and add to list of blocks which should be redistributed
                        for block in sat.sendBufferGT[1]:
                            blocksToDistribute.append((block.checkPoints[-1], block))  # (latest checkpoint time, block)
                        sat.sendBufferGT = ([sat.env.event()], [])

                        # make new send process for new linked GT
                        sat.sendBlocksGT.append(sat.env.process(sat.sendBlock((sat.GTDist, sat.linkedGT), False)))

                else:  # no linked GT
                    # check if there is a sendBlocksGT process
                    if sat.sendBlocksGT:
                        sat.sendBlocksGT[0].interrupt()
                        sat.sendBlocksGT = []

                        # remove blocks from queue and add to list of blocks which should be redistributed
                        for block in sat.sendBufferGT[1]:
                            blocksToDistribute.append((block.checkPoints[-1], block))  # (latest checkpoint time, block)
                        sat.sendBufferGT = ([sat.env.event()], [])

                # sort blocks by arrival time at satellite
                blocksToDistribute.sort()
                # add blocks to the correct queues based on next step in their path
                # since the blocks list is sorted by arrival time, the order in the new queues is correct
                for block in blocksToDistribute:
                    # get this satellite's index in the blocks path
                    index = None
                    for i, step in enumerate(block[1].path):
                        if sat.ID == step[0]:
                            index = i

                    # check if next step in path is GT (last step in path)
                    if index == len(block[1].path) - 2:
                        # add block to GT send-buffer
                        if not sat.sendBufferGT[0][0].triggered:
                            sat.sendBufferGT[0][0].succeed()
                            sat.sendBufferGT[1].append(block[1])
                        else:
                            newEvent = sat.env.event().succeed()
                            sat.sendBufferGT[0].append(newEvent)
                            sat.sendBufferGT[1].append(block[1])
                    else:
                        # get ID of next sat and find if it is intra or inter
                        ID = None
                        isIntra = False
                        for neighborSat in sat.intraSats:
                            id = neighborSat[1].ID
                            if id == block[1].path[index + 1][0]:
                                ID = neighborSat[1].ID
                                isIntra = True
                        for neighborSat in sat.interSats:
                            id = neighborSat[1].ID
                            if id == block[1].path[index + 1][0]:
                                ID = neighborSat[1].ID

                        if ID is not None:
                            sendBuffer = None
                            # find send-buffer for the satellite
                            if isIntra:
                                for buffer in sat.sendBufferSatsIntra:
                                    if ID == buffer[2]:
                                        sendBuffer = buffer
                            else:
                                for buffer in sat.sendBufferSatsInter:
                                    if ID == buffer[2]:
                                        sendBuffer = buffer

                            # add block to buffer
                            if not sendBuffer[0][0].triggered:
                                sendBuffer[0][0].succeed()
                                sendBuffer[1].append(block[1])
                            else:
                                newEvent = sat.env.event().succeed()
                                sendBuffer[0].append(newEvent)
                                sendBuffer[1].append(block[1])
                        else:
                            print("buffer for next satellite in path could not be found")

    def updateGTPaths(self):
        """
        Updates all paths for all GTs going to all other GTs and ensures that all blocks waiting to be sent has the
        correct path.
        用于 更新所有地面站(GT, Gateway)到其他地面站的通信路径 ,并确保地面站发送缓冲区中待传输的数据块使用最新的正确路径
        用来拓扑变化后 动态调整地面站间的路由路径
        """
        # make new paths for all GTs
        for GT in self.gateways:
            for destination in self.gateways:
                if GT != destination:
                    if destination.linkedSat[0] is not None and GT.linkedSat[0] is not None:
                        path = getShortestPath(GT.name, destination.name, self.pathParam, GT.graph)
                        GT.paths.update({destination.name: path})
                    else:
                        GT.paths.update({destination.name: []})

            # update paths for all blocks in send-buffer
            for block in GT.sendBuffer[1]:
                block.path = GT.paths[block.destination.name]
                block.isNewPath = True

    def getGSLDataRates(self):
        """
            收集卫星网络中地面站与卫星间的上下行数据速率(GSL, Ground-Satellite Link)
            返回上行(地面站→卫星)和下行(卫星→地面站)的所有有效数据速率列表
            为后续网络性能分析(如吞吐量统计、链路质量评估)提供基础数据
        """
        upDataRates = []
        downDataRates = []
        for GT in self.gateways:
            if GT.linkedSat[0] is not None:
                upDataRates.append(GT.dataRate)

        for orbit in self.LEO:
            for satellite in orbit.sats:
                if satellite.linkedGT is not None:
                    downDataRates.append(satellite.downRate)

        return upDataRates, downDataRates

    def getISLDataRates(self):
        """
            收集卫星网络中星间链路(ISL, Inter-Satellite Link)的数据速率 
            并统计其中超过3Gbps的高速链路数量,最终返回所有星间链路的数据速率列表
            主要用于卫星网络性能分析(如链路带宽分布、高速链路占比等)
        """
        interDataRates = []
        highRates = 0
        for orbit in self.LEO:
            for satellite in orbit.sats:
                for satData in satellite.interSats:
                    if satData[2] > 3e9:
                        highRates += 1
                    interDataRates.append(satData[2])
        return interDataRates

    def getFlowValues(self):
        """
            用于 计算并保存卫星网络中所有地面站(Gateway)之间的数据传输速率矩阵
            最终将结果存储为CSV文件
            为网络流量分析(如吞吐量分布、链路负载)提供基础数据
        """
        block = DataBlock(self, self.gateways[0], str(1) + "_" + str(1) + "_" + str(1), 1)
        rates = np.ndarray((len(self.gateways), len(self.gateways)))
        for transmitterId, transmitter in enumerate(self.gateways):
            for receiverId, receiver in enumerate(self.gateways):
                if receiver != transmitter:
                    _, flow = transmitter.timeToFullBlock(block, "totalNumb", None)
                    rates[transmitterId, receiverId] = flow
                else:
                    rates[transmitterId, receiverId] = 0
        np.savetxt("./Example_Code/SimulationRates.csv", rates, delimiter=",")

    def setReceiveFractions(self):
        """
        Currently unused, and should be likely be reworked if needed. The current implementation does not diveide
        resources evenly or fairly.

        This function creates the table used for dividing the generated data to the different receiver gts. This
        function must be called after each gt has had its data rate calculated, so we can ensure that we are adhering
        to flow constraints for uplink and downlink.
        用于 生成数据分配比例表 
        目的是将生成地面站( generatorGTs )产生的数据按比例分配到接收地面站( receiverGTs )
        同时确保接收地面站的下行链路(卫星→地面站)不会过载
        明确标注该方法 当前未使用
        """
        fractions = np.zeros([len(self.generatorGTs), len(self.receiverGTs)])

        traffic = [0 for _ in range(len(self.receiverGTs))]

        # fill fractions table
        for transmitterIndex, transmitter in enumerate(self.generatorGTs):
            for receiverIndex, receiver in enumerate(self.receiverGTs):
                # ensure that downlink is not overloaded
                if traffic[receiverIndex] + transmitter.totalAvgFlow * (1/(len(self.receiverGTs))) > receiver.linkedSat[1].downRate:

                    frac = (receiver.linkedSat[1].downRate - traffic[receiverIndex]) / transmitter.totalAvgFlow

                    fractions[transmitterIndex, receiverIndex] = frac

                    traffic[receiverIndex] += receiver.linkedSat[1].downRate - traffic[receiverIndex]
                else:
                    traffic[receiverIndex] += transmitter.totalAvgFlow * (1/(len(self.receiverGTs)))
                    fractions[transmitterIndex, receiverIndex] = 1/(len(self.receiverGTs))

        for receiverIndex, receiver in enumerate(self.receiverGTs):
            if traffic[receiverIndex] > receiver.linkedSat[1].downRate:
                print("error")
                print(receiver)
                print(receiver.linkedSat[1].downRate)
                print(traffic[receiverIndex])

        self.fractions = fractions

    def moveConstellation(self, env, deltaT=3600, getRates = False):
        """
        Simpy process function:

        Moves the constellations in terms of the Earth's rotation and moves the satellites within the constellations.
        The movement is based on the time that has passed since last constellation movement and is defined by the
        "deltaT" variable.

        After the satellites have been moved a process of re-linking all links, both GSLs and ISLs, is conducted where
        the paths for all blocks are re-made, the blocks are moved (if necessary) to the correct buffers, and all
        processes managing the send-buffers are checked to ensure they will still work correctly.
        
        用于 动态移动卫星星座并更新网络拓扑 的核心Simpy进程函数
        核心作用是模拟地球自转和卫星轨道运动导致的星座位置变化
        通过重新 建立链路 更新路径和传输进程, 确保卫星网络通信的连续性
        """

        # Get the data rate for a intra plane ISL - used for testing
        if getRates:
            intraRate.append(self.LEO[0].sats[0].intraSats[0][2])

        while True:
            if getRates:
                # get data rates for all inter plane ISLs and all GSLs (up and down) - used for testing
                upDataRates, downDataRates = self.getGSLDataRates()
                inter = self.getISLDataRates()

                for val in upDataRates:
                    upGSLRates.append(val)

                for val in downDataRates:
                    downGSLRates.append(val)

                for val in inter:
                    interRates.append(val)

            yield env.timeout(deltaT)

            # clear satellite references on all GTs
            for GT in self.gateways:
                GT.satsOrdered = []
                GT.linkedSat = (None, None)

            # rotate constellation and satellites
            for constellation in self.LEO:
                constellation.rotate(deltaT)

            # relink satellites and GTs
            self.linkSats2GTs("Optimize")

            # create new graph and add references to all GTs for every rotation
            graph = createGraph(self)
            self.graph = graph
            for GT in self.gateways:
                GT.graph = graph

            self.updateSatelliteProcessesCorrect(graph)

            self.updateGTPaths()

    def testFlowConstraint1(self, graph):
        """
            验证卫星网络拓扑图中链路距离是否符合流量约束条件
            核心逻辑是通过计算地面站连接卫星的最小距离阈值, 统计拓扑图中超过该阈值的边的数量, 辅助判断网络链路的有效性
        """
        highestDist = (0,0)
        for GT in self.gateways:
            if 1/GT.linkedSat[0] > highestDist[0]:
                highestDist = (1/GT.linkedSat[0], GT)

        lowestDist = (1/highestDist[0], highestDist[1])

        toolargeDists = []

        for (u,v,c) in graph.edges.data("slant_range"):
            if c > lowestDist[0]:
                toolargeDists.append((u,v,c))

        print("number of edges with too large distance: {}".format(len(toolargeDists)))

    def testFlowConstraint2(self, graph):
        """
            验证卫星网络中地面站间路径是否符合流量约束条件
                过验证地面站间路径的距离约束
                确保卫星网络中的流量传输路径在物理层(如信号强度、传输延迟)上的合理性
                避免因后续路径段过长导致的传输质量下降或链路中断
                是网络路由算法验证的重要辅助逻辑
        """
        edgeWeights = nx.get_edge_attributes(graph, "slant_range")
        totalFailed = 0

        for GT in self.gateways[1:]:
            failed = False
            path = getShortestPath(self.gateways[0].name, GT.name, 'slant_range', graph)
            try:
                firstStep = GT.linkedSat[0]
            except KeyError:
                firstStep = edgeWeights[(path[1][0], path[0][0])] 

            for index in range(1, len(path) - 2):
                try:
                    if edgeWeights[(path[index][0], path[index+1][0])] > firstStep:
                        failed = True
                except KeyError:
                    if edgeWeights[(path[index+1][0], path[index][0])] > firstStep:
                        failed = True
            if failed:
                print("{} could not create a path which adheres to flow constraints".format(GT.name))
                totalFailed += 1

        print("number of GT paths that cannot meet flow restraints: {}".format(totalFailed))

    def plotMap(self, plotGT = True, plotSat = True, path = None, bottleneck = None):
        """
            用于 可视化卫星网络的全局拓扑. 支持绘制地面站(GT),卫星(Sat)的位置, 以及可选的通信路径(path)和瓶颈链路(bottleneck)
            核心作用是通过地图形式直观展示卫星网络的分布、连接关系及关键路径，辅助分析网络覆盖和路由性能
        """
        plt.figure()
        if plotGT:
            for GT in self.gateways:
                scat1 = plt.scatter(GT.gridLocationX, GT.gridLocationY, marker='x', c='r', s=28, linewidth=1.5, label = GT.name)
                # if GT.linkedSat[0] is not None:
                    # scat1 = plt.scatter(GT.gridLocationX, GT.gridLocationY, marker='x', c='r', s=8, linewidth=0.5)
                    # gridSatX = int((0.5 + math.degrees(GT.linkedSat[1].longitude) / 360) * GT.totalX)
                    # gridSatY = int((0.5 - math.degrees(GT.linkedSat[1].latitude) / 180) * GT.totalY)
                    #scat2 = plt.scatter(gridSatX, gridSatY, marker='o', s=8, linewidth=0.5, color='r')
                    # print(GT.linkedSat[1])
                    # print(GT)

            colors = matplotlib.cm.rainbow(np.linspace(0, 1, len(self.LEO)))
            
        if plotSat:    
            for plane, c in zip(self.LEO, colors):
                # print('------------------------------------------------------------')
                # print('Plane: ' + str(plane.ID))
                for sat in plane.sats:
                    gridSatX = int((0.5 + math.degrees(sat.longitude) / 360) * 1440)
                    gridSatY = int((0.5 - math.degrees(sat.latitude) / 180) * 720) #GT.totalY)
                    scat2 = plt.scatter(gridSatX, gridSatY, marker='o', s=18, linewidth=0.5, color=c, label = sat.ID)
                    # print('Longitude: ' + str(math.degrees(sat.longitude)) +  ', Grid X: ' + str(gridSatX) + '\nLatitude: ' + str(math.degrees(sat.latitude)) + ', Grid Y: ' + str(gridSatY))
                        # Longitude +-180º, latitude +-90º

        # Print path if given
        if path:
            # print('Plotting path between ' + path[0][0] + ' and ' + path[len(path)-1][0])
            if bottleneck:
                xValues = [[], [], []]
                yValues = [[], [], []]
                # bottleneck[1][-1] = 1 # used to test all links to ensure code is correct in plotting path and weakest link
                minimum = np.amin(bottleneck[1])
                length = len(path)
                index = 0
                arr = 0
                minFound = False

                while index < length:
                    xValues[arr].append(int((0.5 + path[index][1] / 360) * 1440))  # longitude
                    yValues[arr].append(int((0.5 - path[index][2] / 180) * 720))  # latitude
                    if not minFound:
                        if bottleneck[1][index] == minimum:
                            arr+=1
                            xValues[arr].append(int((0.5 + path[index][1] / 360) * 1440))  # longitude
                            yValues[arr].append(int((0.5 - path[index][2] / 180) * 720))  # latitude
                            xValues[arr].append(int((0.5 + path[index+1][1] / 360) * 1440))  # longitude
                            yValues[arr].append(int((0.5 - path[index+1][2] / 180) * 720))  # latitude
                            arr+=1
                            minFound = True
                    index += 1

                scat3 = plt.plot(xValues[0], yValues[0], 'b')
                scat3 = plt.plot(xValues[1], yValues[1], 'r')
                scat3 = plt.plot(xValues[2], yValues[2], 'b')
            else:
                xValues = []
                yValues = []
                for hop in path:
                    xValues.append(int((0.5 + hop[1] / 360) * 1440))     # longitude
                    yValues.append(int((0.5 - hop[2] / 180) * 720))      # latitude
                scat3 = plt.plot(xValues, yValues)  # , marker='.', c='b', linewidth=0.5, label = hop[0])
            
            # plt.legend([scat1, scat2, scat3], ['Ground Terminals', 'Satellites', 'Path'], loc=3, prop={'size': 7})
        
        if plotSat and plotGT:
            plt.legend([scat1, scat2], ['Concentrators', 'Satellites'], loc=3, prop={'size': 7})
        elif plotSat:
            plt.legend([scat2], ['Satellites'], loc=3, prop={'size': 7})
        elif plotGT:
            plt.legend([scat1], ['Concentrators'], loc=3, prop={'size': 7})

        plt.xticks([])
        plt.yticks([])


        plt.imshow(np.log10(np.array(self.getCellUsers()).transpose() + 1), )
        # plt.title('LEO constellation and Ground Terminals')
        # plt.rcParams['figure.figsize'] = 36, 12  # adjust if figure is too big or small for screen
        # plt.colorbar(fraction=0.1)  # adjust fraction to change size of color bar
        # plt.show()

    def plot3D(self):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        xs = []
        ys = []
        zs = []
        xG = []
        yG = []
        zG = []
        for con in self.LEO:
            for sat in con.sats:
                xs.append(sat.x)
                ys.append(sat.y)
                zs.append(sat.z)
        ax.scatter(xs, ys, zs, marker='o')
        for GT in self.gateways:
            xG.append(GT.x)
            yG.append(GT.y)
            zG.append(GT.z)
        ax.scatter(xG, yG, zG, marker='^')
        plt.show()

    def __repr__(self):
        return 'total divisions in x = {}\n total divisions in y = {}\n total cells = {}\n window of operation ' \
               '(longitudes) = {}\n window of operation (latitudes) = {}'.format(
                self.total_x,
                self.total_y,
                self.total_cells,
                self.windowx,
                self.windowy)


###############################################################################
###############################    Functions    ###############################
###############################################################################


"""
    卫星网络仿真的 核心初始化函数
    负责完成仿真前的所有关键准备工作
    包括加载地理数据、构建卫星星座、建立地面站与卫星的连接、生成网络拓扑图、规划通信路径，以及初始化数据传输的缓冲区和进程
    最终返回仿真所需的核心对象

    关键作用为:
        1. 物理拓扑构建: 建立地面站与卫星的连接, 生成星间链路和地面链路的网络拓扑
        2. 路径规划: 预计算所有地面站对的最短路径, 确保数据传输的高效性
        3. 传输初始化: 为卫星和地面站创建缓冲区和发送进程, 支撑后续的实时数据传输仿真
        4. 流量控制: 通过识别瓶颈链路和设置流量生成策略, 避免网络过载, 保证仿真的真实性
"""
def initialize(env, popMapLocation, GTLocation, distance, inputParams, movementTime, totalLocations):
    """
    Initializes an instance of the earth with cells from a population map and gateways from a csv file.
    During initialisation, several steps are performed to prepare for simulation:
        - GTs find the cells that within their ground coverage areas and "link" to them.
        - A certain LEO Constellation with a given architecture is created.
        - Satellites are distributed out to GTs so each GT connects to one satellite (if possible) and each satellite
        only has one connected GT.
        - A graph is created from all the GSLs and ISLs
        - Paths are created from each GT to all other GTs
        - Buffers and processes are created on all GTs and satellites used for sending the blocks throughout the network
    """

    # 参数解析与标志设置
    constellationType = inputParams['Constellation'][0]
    fraction = inputParams['Fraction'][0]
    testType = inputParams['Test type'][0]

    if testType == "Rates":
        getRates = True
    else:
        getRates = False

    # Load earth and gateways
    # 初始化地球与地面站
    earth = Earth(env, popMapLocation, GTLocation, constellationType, inputParams, movementTime, totalLocations, getRates)

    print(earth)
    print()

    # 连接地面站与覆盖区域, 优化卫星与地面站连接, 构建网络拓扑图
    earth.linkCells2GTs(distance)
    earth.linkSats2GTs("Optimize")
    graph = createGraph(earth)

    for gt in earth.gateways:
        gt.graph = graph

    paths = []
    # make paths for all source destination pairs
    # 生成地面站之间的通信路径
    for GT in earth.gateways:
        for destination in earth.gateways:
            if GT != destination:
                if destination.linkedSat[0] is not None and GT.linkedSat[0] is not None:
                    path = getShortestPath(GT.name, destination.name, earth.pathParam, GT.graph)
                    GT.paths[destination.name] = path
                    paths.append(path)

    # add ISl references to all satellites and adjust data rate to GTs
    # 添加轨道卫星到卫星列表
    sats = []
    for plane in earth.LEO:
        for sat in plane.sats:
            sats.append(sat)

    fiveNeighbors = ([0],[])

    pathNames = [name[0] for name in path]

    # 初始化卫星链路与传输进程
    for plane in earth.LEO:
        for sat in plane.sats:

            if sat.linkedGT is not None:
                sat.adjustDownRate()    # 调整卫星下行速率
                # make a process for the GSL from sat to GT
                sat.sendBlocksGT.append(sat.env.process(sat.sendBlock((sat.GTDist, sat.linkedGT), False)))  # 创建GSL发送进程
            neighbors = list(nx.neighbors(graph, sat.ID))   # 获取卫星的星间链路邻居
            if len(neighbors) == 5:
                fiveNeighbors[0][0] += 1
                fiveNeighbors[1].append(neighbors)
            itt = 0
            for sat2 in sats:

                if sat2.ID in neighbors:
                    dataRate = nx.path_weight(graph,[sat2.ID, sat.ID], "dataRateOG")    # 获取链路数据速率
                    distance = nx.path_weight(graph,[sat2.ID, sat.ID], "slant_range")   # 获取链路距离

                    # check if satellite is inter- or intra-plane
                    # 区分同轨面（intra）和异轨面（inter）链路
                    if sat2.in_plane == sat.in_plane:
                        sat.intraSats.append((distance, sat2, dataRate))                                                        # 同轨面链路列表
                        # make a send buffer for intra ISL ([self.env.event()], [DataBlock(0, 0, "0", 0)], 0)
                        sat.sendBufferSatsIntra.append(([sat.env.event()], [], sat2.ID))                                        # 同轨面发送缓冲区
                        # make a process for intra ISL
                        sat.sendBlocksSatsIntra.append(sat.env.process(sat.sendBlock((distance, sat2, dataRate), True, True)))  # 同轨面发送进程
                    else:
                        sat.interSats.append((distance, sat2, dataRate))                                                        # 异轨面链路列表
                        # make a send buffer for inter ISL ([self.env.event()], [DataBlock(0, 0, "0", 0)], 0)
                        sat.sendBufferSatsInter.append(([sat.env.event()], [], sat2.ID))                                        # 异轨面发送缓冲区
                        # make a process for inter ISL
                        sat.sendBlocksSatsInter.append(sat.env.process(sat.sendBlock((distance, sat2, dataRate), True, False))) # 异轨面发送进程

                    itt += 1
                    if itt == len(neighbors):
                        break

    # 调用 findBottleneck 函数, 识别前两条路径的瓶颈链路 (即路径中数据速率最小的链路), 用于后续流量控制
    bottleneck2, minimum2 = findBottleneck(paths[1], earth, False)
    bottleneck1, minimum1 = findBottleneck(paths[0], earth, False, minimum2)

    # 初始化地面站流量生成
    for GT in earth.gateways:
        mins = []
        if GT.linkedSat[0] is not None:
            for pathKey in GT.paths:
                _, minimum = findBottleneck(GT.paths[pathKey], earth)
                mins.append(minimum)    # 收集所有路径的最小带宽

            # 根据地面站数据速率与卫星下行速率的比较, 选择流量生成策略. 优先使用数据速率较小的链路, 避免链路过载
            if GT.dataRate < GT.linkedSat[1].downRate:
                GT.getTotalFlow(1, "Step", 1, GT.dataRate, fraction)  # using data rate of the GSL uplink   使用GSL上行速率
            else:
                GT.getTotalFlow(1, "Step", 1, GT.linkedSat[1].downRate, fraction)  # using data rate of the GSL downlink    使用GSL下行速率

            # alternative initializers for the generated flow:
            # GT.getTotalFlow(1, "Step", 1, np.amin(mins), fraction) # using the data rate of the path capacity
            # GT.getTotalFlow(1, "Step", 1, GT.linkedSat[1].downRate, fraction) # using data rate of the GSL downlink
            # GT.getTotalFlow(1, "Step", 1, GT.dataRate, fraction)  # using data rate of the GSL uplink

    # ensure fraction table is created and can adhere to constraints given uplink and downlink.
    # the fractions table is currently unused but is needed if select generators and receivers are used in Earth innit
    # earth.setReceiveFractions()

    return earth, graph, bottleneck1, bottleneck2


"""
    用于 查找给定路径中的瓶颈链路
    核心功能是遍历路径上的所有链路, 并记录每条链路的带宽, 最后返回带宽最小的链路及其带宽值
    主要用于网络分析和优化, 帮助确定网络中的瓶颈链路, 从而进行流量控制和资源分配
"""
def findBottleneck(path, earth, plot = False, minimum = None):
    # Find the bottleneck of a route.
    bottleneck = [[], [], [], []]
    for GT in earth.gateways:
        if GT.name == path[0][0]:
            bottleneck[0].append(str(path[0][0].split(",")[0]) + "," + str(path[1][0]))
            bottleneck[1].append(GT.dataRate)
            bottleneck[2].append(GT.latitude)
            if minimum:
                bottleneck[3].append(minimum/GT.dataRate)

    for i, step in enumerate(path[1:], 1):
        for orbit in earth.LEO:
            for satellite in orbit.sats:
                if satellite.ID == step[0]:

                    for sat in satellite.interSats:
                        if sat[1].ID == path[i + 1][0]:
                            bottleneck[0].append(str(path[i][0]) + "," + str(path[i + 1][0]))
                            bottleneck[1].append(sat[2])
                            bottleneck[2].append(satellite.latitude)
                            if minimum:
                                bottleneck[3].append(minimum / sat[2])
                    for sat in satellite.intraSats:
                        if sat[1].ID == path[i + 1][0]:
                            bottleneck[0].append(str(path[i][0]) + "," + str(path[i + 1][0]))
                            bottleneck[1].append(sat[2])
                            bottleneck[2].append(satellite.latitude)
                            if minimum:
                                bottleneck[3].append(minimum / sat[2])
    for GT in earth.gateways:
        if GT.name == path[-1][0]:
            bottleneck[0].append(str(path[-2][0]) + "," + str(path[-1][0].split(",")[0]))
            bottleneck[1].append(GT.linkedSat[1].downRate)
            bottleneck[2].append(GT.latitude)
            if minimum:
                bottleneck[3].append(minimum/GT.dataRate)

    if plot:
        earth.plotMap(True,True,path, bottleneck)
        plt.show()

    minimum = np.amin(bottleneck[1])
    return bottleneck, minimum


"""
    用于 创建卫星星座
    主要功能是根据指定的星座类型和参数, 生成卫星网络的拓扑结构
    包括轨道平面数(P), 每个轨道平面上的卫星数量(N_p), 卫星的高度(height), 轨道平面的倾斜角(inclination_angle), 卫星分布角度(distribution_angle), 以及是否采用Walker星型分布(Walker_star)
"""
def create_Constellation(specific_constellation, env):

    if specific_constellation == "small":               # Small Walker star constellation for tests.
        print("Using small walker Star constellation")
        P = 3					# Number of orbital planes
        N_p = 4 				# Number of satellites per orbital plane
        N = N_p*P				# Total number of satellites
        height = 1000e3			# Altitude of deployment for each orbital plane (set to the same altitude here)
        inclination_angle = 53	# Inclination angle for the orbital planes, set to 90 for Polar 
        Walker_star = True		# Set to True for Walker star and False for Walker Delta
        min_elevation_angle = 30

    elif specific_constellation =="Kepler":
        print("Using Kepler constellation design")
        P = 7
        N_p = 20
        N = N_p*P
        height = 600e3
        inclination_angle = 98.6
        Walker_star = True
        min_elevation_angle = 30

    elif specific_constellation =="Iridium_NEXT":
        print("Using Iridium NEXT constellation design")
        P = 6
        N_p = 11
        N = N_p*P
        height = 780e3
        inclination_angle = 86.4
        Walker_star = True
        min_elevation_angle = 30

    elif specific_constellation =="OneWeb":
        print("Using OneWeb constellation design")
        P = 18
        N = 648	
        N_p = int(N/P)
        height = 1200e3
        inclination_angle = 86.4
        Walker_star = True
        min_elevation_angle = 30

    elif specific_constellation =="Starlink":			# Phase 1 550 km altitude orbit shell
        print("Using Starlink constellation design")
        P = 72
        N = 1584
        N_p = int(N/P)
        height = 550e3
        inclination_angle = 53
        Walker_star = False
        min_elevation_angle = 25

    elif specific_constellation == "Test":
        print("Using a test constellation design")
        P = 30                     # Number of orbital planes
        N = 1200                   # Total number of satellites
        N_p = int(N/P)             # Number of satellites per orbital plane
        height = 600e3             # Altitude of deployment for each orbital plane (set to the same altitude here)
        inclination_angle = 86.4   # Inclination angle for the orbital planes, set to 90 for Polar
        Walker_star = True         # Set to True for Walker star and False for Walker Delta
        min_elevation_angle = 30
    else:
        print("Not valid Constellation Name")
        P = np.NaN
        N_p = np.NaN
        N = np.NaN
        height = np.NaN
        inclination_angle = np.NaN
        Walker_star = False
        exit()
    
    distribution_angle = 2*math.pi  # Angle in which the orbital planes are distributed in
    
    if Walker_star:
        distribution_angle /= 2
    orbital_planes = []

    # Add orbital planes and satellites
    # Orbital_planes.append(orbital_plane(0, height, 0, math.radians(inclination_angle), N_p, min_elevation_angle, 0))
    for i in range(0, P):
        orbital_planes.append(OrbitalPlane(str(i), height, i*distribution_angle/P, math.radians(inclination_angle), N_p,
                                           min_elevation_angle, str(i) + '_', env))

    return orbital_planes

###############################################################################
###############################  Create Graph   ###############################
###############################################################################


def get_direction(Satellites):
    '''
    Gets the direction of the satellites so each transceiver antenna can be set to one direction.
    '''
    N = len(Satellites)
    direction = np.zeros((N,N), dtype=np.int8) 
    for i in range(N):
        epsilon = -Satellites[i].inclination    # orbital plane inclination
        for j in range(N):
            direction[i,j] = np.sign(Satellites[i].y*math.sin(epsilon)+ 
                                    Satellites[i].z*math.cos(epsilon)-Satellites[j].y*math.sin(epsilon)- 
                                    Satellites[j].z*math.cos(epsilon))
    return direction


def get_pos_vectors_omni(Satellites):
    '''
    Given a list of satellites returns a list with x, y, z coordinates and the plane where they are (meta)
    '''
    N = len(Satellites)
    Positions = np.zeros((N,3))
    meta = np.zeros(N, dtype=np.int_)
    for n in range(N):
        #Satellites[n].rotate_axes([1,0,0],-Orbital_planes[1].inclination)
        Positions[n,:] = [Satellites[n].x, Satellites[n].y, Satellites[n].z]
        meta[n] = Satellites[n].in_plane
    
    return Positions, meta


def get_slant_range(edge):
        return(edge.slant_range)


@numba.jit  # Using this decorator you can mark a function for optimization by Numba's JIT compiler
def get_slant_range_optimized(Positions, N):
    '''
    returns a matrix with the all the distances between the satellites (optimized)
    '''
    slant_range = np.zeros((N,N))
    for i in range(N):
        slant_range[i,i] = math.inf
        for j in range(i+1,N):
            slant_range[i,j] = np.linalg.norm(Positions[i,:] - Positions[j,:])
    slant_range += np.transpose(slant_range)
    return slant_range


@numba.jit  # Using this decorator you can mark a function for optimization by Numba's JIT compiler
def los_slant_range(_slant_range, _meta, _max, _Positions):
    ''' 
    line of sight slant range
    '''
    _slant_range_new = np.copy(_slant_range)
    _N = len(_slant_range)
    for i in range(_N):
        for j in range(_N):
            if _slant_range_new[i,j] > _max[_meta[i], _meta[j]]:
                _slant_range_new[i,j] = math.inf
    return _slant_range_new


def get_data_rate(_slant_range_los, interISL):
    """
    Given a matrix of slant ranges returns a matrix with all the shannon dataRates possibles between all the satellites.
    """
    speff_thresholds = np.array(
        [0, 0.434841, 0.490243, 0.567805, 0.656448, 0.789412, 0.889135, 0.988858, 1.088581, 1.188304, 1.322253,
         1.487473, 1.587196, 1.647211, 1.713601, 1.779991, 1.972253, 2.10485, 2.193247, 2.370043, 2.458441,
         2.524739, 2.635236, 2.637201, 2.745734, 2.856231, 2.966728, 3.077225, 3.165623, 3.289502, 3.300184,
         3.510192, 3.620536, 3.703295, 3.841226, 3.951571, 4.206428, 4.338659, 4.603122, 4.735354, 4.933701,
         5.06569, 5.241514, 5.417338, 5.593162, 5.768987, 5.900855])
    lin_thresholds = np.array(
        [1e-10, 0.5188000389, 0.5821032178, 0.6266138647, 0.751622894, 0.9332543008, 1.051961874, 1.258925412,
         1.396368361, 1.671090614, 2.041737945, 2.529297996, 2.937649652, 2.971666032, 3.25836701, 3.548133892,
         3.953666201, 4.518559444, 4.83058802, 5.508076964, 6.45654229, 6.886522963, 6.966265141, 7.888601176,
         8.452788452, 9.354056741, 10.49542429, 11.61448614, 12.67651866, 12.88249552, 14.48771854, 14.96235656,
         16.48162392, 18.74994508, 20.18366364, 23.1206479, 25.00345362, 30.26913428, 35.2370871, 38.63669771,
         45.18559444, 49.88844875, 52.96634439, 64.5654229, 72.27698036, 76.55966069, 90.57326009])

    pathLoss = 10*np.log10((4 * math.pi * _slant_range_los * interISL.f / Vc)**2)   # Free-space pathloss in dB
    snr = 10**((interISL.maxPtx_db + interISL.G - pathLoss - interISL.No)/10)       # SNR in times
    shannonRate = interISL.B*np.log2(1+snr)                                         # data rates matrix in bits per second

    speffs = np.zeros((len(_slant_range_los),len(_slant_range_los)))

    for n in range(len(_slant_range_los)):
        for m in range(len(_slant_range_los)):
            feasible_speffs = speff_thresholds[np.nonzero(lin_thresholds <= snr[n,m])]
            if feasible_speffs.size == 0:
                speffs[n, m] = 0
            else:
                speffs[n,m] = interISL.B * feasible_speffs[-1]

    return speffs


def markovianMatchingTwo(earth):
    '''
    Returns a list of edge class elements. Each edge stands for a connection between two satellites. On that class
    the slant range and the data rate between both satellites are stored as attributes.
    This function is for satellites with two transceivers antennas that will enable two inter-plane ISL each one
    in a different direction.
    Intra-plane ISL are also computed and returned in _A_Markovian list

    It is not the optimal solution, but it is from 10 to 1000x faster.
    Minimizes the total cost of the constellation matching problem.
    '''

    _A_Markovian    = []    # list with all the
    Satellites      = []    # list with all the satellites
    W_M             = []    # list with the distances of every possible link between sats
    covered         = set() # Set with the connections already covered
    
    for plane in earth.LEO:
        for sat in plane.sats:
            Satellites.append(sat)

    N = len(Satellites)

    interISL = RFlink(
        frequency=26e9,
        bandwidth=500e6,
        maxPtx=10,
        aDiameterTx=0.26,
        aDiameterRx=0.26,
        pointingLoss=0.3,
        noiseFigure=2,
        noiseTemperature=290,
        min_rate=10e3
    )

    # max slant range for each orbit
    ###########################################################
    M = len(earth.LEO)              # Number of planes in LEO
    Max_slnt_rng = np.zeros((M,M))  # All ISL slant ranges must me lowe than 'Max_slnt_rng[i, j]'

    Orb_heights  = []
    for plane in earth.LEO:
        Orb_heights.append(plane.h)
        maxSlantRange = plane.sats[0].maxSlantRange()

    for _i in range(M):
        for _j in range(M):
            Max_slnt_rng[_i,_j] = (np.sqrt( (Orb_heights[_i] + Re)**2 - Re**2 ) +
                                np.sqrt( (Orb_heights[_j] + Re)**2 - Re**2 ) )


    # Get data rate old method
    ###########################################################
    direction       = get_direction(Satellites)             # get both directions of the satellites to use the two transceivers
    Positions, meta = get_pos_vectors_omni(Satellites)      # position and plane of all the satellites
    slant_range     = get_slant_range_optimized(Positions, N)                       # matrix with all the distances between satellties
    slant_range_los = los_slant_range(slant_range, meta, Max_slnt_rng, Positions)   # distance matrix but if d>dMax, d=infinite
    shannonRate     = get_data_rate(slant_range_los, interISL)                      # max dataRate

    '''
    Compute all possible edges between different plane satellites whose transceiver antennas are free.
    if slant range > max slant range then that edge is not added
    '''
    ###########################################################
    for i in range(N):
        for j in range(i+1,N):
            if Satellites[i].in_plane != Satellites[j].in_plane and ((i,direction[i,j]) not in covered) and ((j,direction[j,i]) not in covered):
                if slant_range_los[i,j] < 6000e3: # math.inf:
                    W_M.append(edge(Satellites[i].ID,Satellites[j].ID,slant_range_los[i,j],direction[i,j], direction[j,i],     shannonRate[i,j]))
    
    W_sorted=sorted(W_M,key=get_slant_range)
    
    # from all the possible links adds only the uncovered with the best weight possible
    ###########################################################
    while W_sorted: 
        if  ((W_sorted[0].i,W_sorted[0].dji) not in covered) and ((W_sorted[0].j,W_sorted[0].dij) not in covered):
            _A_Markovian.append(W_sorted[0])
            covered.add((W_sorted[0].i,W_sorted[0].dji))
            covered.add((W_sorted[0].j,W_sorted[0].dij))
        W_sorted.pop(0)
    
    # add intra-ISL edges
    ###########################################################
    nPlanes = len(earth.LEO)
    for plane in earth.LEO:
        nPerPlane = len(plane.sats)
        for sat in plane.sats:
            sat.findNeighbours(earth)

            # upper neighbour
            i = sat.in_plane        *nPerPlane    +sat.i_in_plane

            j = sat.upper.in_plane  *nPerPlane    +sat.upper.i_in_plane

            _A_Markovian.append(edge(sat.ID, sat.upper.ID,  # satellites IDs
            slant_range_los[i, j],                          # distance between satellites
            direction[i,j], direction[j,i],                 # directions
            shannonRate[i,j]))                              # Max dataRate

            # lower neighbour
            j = sat.lower.in_plane  *nPerPlane    +sat.lower.i_in_plane

            _A_Markovian.append(edge(sat.ID, sat.lower.ID,  # satellites IDs
            slant_range_los[i, j],                          # distance between satellites
            direction[i,j], direction[j,i],                 # directions
            shannonRate[i,j]))                              # Max dataRate

    return _A_Markovian


def createGraph(earth):
    '''
    Each satellite has two transceiver antennas that are connected to the closest satellite in east and west direction to a satellite
    from another plane (inter-ISL). Each satellite also has anoteher two transceiver antennas connected to the previous and to the 
    following satellite at their orbital plane (intra-ISL). 
    A graph is created where each satellite is a node and each connection is an edge with a specific weight based either on the 
    inverse of the maximum data rate achievable, total distance or number of hops. 
    '''
    g = nx.Graph()

    # add LEO constellation
    ###############################
    for plane in earth.LEO:
        for sat in plane.sats:
            g.add_node(sat.ID, sat=sat)

    # add gateways and GSL edges
    ###############################
    for GT in earth.gateways:
        if GT.linkedSat[1]:
            g.add_node(GT.name, GT = GT)            # add GT as node
            g.add_edge(GT.name, GT.linkedSat[1].ID, # add GT linked sat as edge
            slant_range = GT.linkedSat[0],          # slant range
            invDataRate = 1/GT.dataRate,            # Inverse of dataRate
            dataRateOG = GT.dataRate,               # original shannon dataRate
            hop = 1,                                # in case we just want to count hops
            latency = 1)                            # the latency is based on the current queue size plus 1, the data rate, and the length of the link
        # else:
            # print(GT.name + ' has no linked satellite :(')


    # add inter-ISL and intra-ISL edges
    ###############################
    markovEdges = markovianMatchingTwo(earth)
    for markovEdge in markovEdges:
        g.add_edge(markovEdge.i, markovEdge.j,  # source and destination IDs
        slant_range = markovEdge.slant_range,   # slant range
        dataRate = 1/markovEdge.shannonRate,    # Inverse of dataRate
        dataRateOG = markovEdge.shannonRate,    # Original shannon datRate
        hop = 1,                                # in case we just want to count hops
        latency = 1)

    return g


def getShortestPath(source, destination, weight, g):
    '''
    Gives you the shortest path between a source and a destination and plots it if desired.
    Uses the 'dijkstra' algorithm to compute the sortest path, where the total weight of the path can be either the sum of inverse
    of the maximumm dataRate achevable, the total slant range or the number of hops taken between source and destination.

    returns a list where each element is a sublist with the name of the node, its longitude and its latitude.
    '''

    path = []
    try:
        shortest = nx.shortest_path(g, source, destination, weight = weight)    # computes the shortest path [dataRate, slant_range, hops]
        for hop in shortest:                                                    # pre process the data so it can be used in the future
            key = list(g.nodes[hop])[0]
            if shortest.index(hop) == 0 or shortest.index(hop) == len(shortest)-1:
                path.append([hop, g.nodes[hop][key].longitude, g.nodes[hop][key].latitude])
            else:
                path.append([hop, math.degrees(g.nodes[hop][key].longitude), math.degrees(g.nodes[hop][key].latitude)])
    except:
        print('No path between ' + source + ' and ' + destination + ', check the graph to see more details.')
        return -1
    return path


def plotShortestPath(earth, path):
    earth.plotMap(True, True, path=path)
    plt.savefig('popMap_{}.pdf'.format(500), dpi = 500)
    plt.show()


def normalize(arr, t_min, t_max):
    norm_arr = []
    diff = t_max - t_min
    diff_arr = max(arr) - min(arr)
    for i in arr:
        temp = (((i - min(arr))*diff)/diff_arr) + t_min
        norm_arr.append(temp)
    return norm_arr


def plotLatencies(percentages, pathing, savePath):
    '''
    Bar plot where each bar is a scenario with a different nº of gateways and where each color represents one of the three latencies.
    '''
        # plot percent stacked barplot
    barWidth= 0.85
    r       = percentages['GTnumber']
    numbers = percentages['GTnumber']
    GTnumber= len(r)

    plt.bar(r, percentages['Propagation time'], color='#b5ffb9', edgecolor='white', width=barWidth, label="Propagation time")   # Propagation time
    plt.bar(r, percentages['Queue time'], bottom=percentages['Propagation time'], color='#f9bc86',                              # Queue time
             edgecolor='white', width=barWidth, label="Queue time")
    plt.bar(r, percentages['Transmission time'], bottom=[i+j for i,j in zip(percentages['Propagation time'],                    # Tx time
            percentages['Queue time'])], color='#a3acff', edgecolor='white', width=barWidth, label="Transmission time")

    # Custom x axis
    plt.xticks(numbers)
    plt.xlabel("Nº of gateways")
    plt.ylabel('Latency')

    # Add a legend
    plt.legend(loc='lower left')
    
    # Show and save graphic
    plt.savefig(
        savePath + '{}_gatewaysTotal.png'.format(GTnumber))

    data = {"numb gateways": r, "prop delay": percentages['Propagation time'], "Queue delay": percentages['Queue time'], "transmission delay": percentages['Transmission time']}
    d = pd.DataFrame(data=data)
    d.to_csv(savePath + "delayFractions.csv")
    # try:
    #     plt.savefig('Latency3/{}/Percentages_{}_gateways.png'.format(pathing, GTnumber))
    # except:
    #     plt.savefig('./Code/Latency3/{}/Percentages_{}_gateways.png'.format(pathing, GTnumber))
    # plt.show()


def main():
    """
    This function is made to avoid problems with scope. everything in if __name__ = "__main__" is in global scope which
    can be an issue.
    """
    percentages = {'Queue time': [],
                   'Propagation time': [],
                   'Transmission time': [],
                   'GTnumber': []}

    inputParams = pd.read_csv("input.csv")

    locations = inputParams['Locations'].copy()
    print(len(locations))

    pathing = inputParams['Pathing'][0] # possible pathing metrics: "slant_range", "dataRate", "hop", "latency".
    testType = inputParams['Test type'][0]
    testLength = inputParams['Test length'][0]

    # movement time should be in the order of 10's of hours when the test type is "Rates".
    # If the test is not 'Rates', the movement time is still kept large to avoid the constellation moving
    movementTime = 10 * 3600

    savePath1 = "./Results/latency Test/{} {}s/".format(pathing, int(testLength))

    if not os.path.exists(savePath1):
        # Create a new directory because it does not exist
        os.makedirs(savePath1)

    savePath2 = savePath1 + "{}/".format(testType)

    if not os.path.exists(savePath2):
        # Create a new directory because it does not exist
        os.makedirs(savePath2)

    blockPath = "./Results/Congestion_test/{} {}s/".format(pathing, int(testLength))
    if not os.path.exists(blockPath):
        # Create a new directory because it does not exist
        os.makedirs(blockPath)

    if testType == "Rates":
        numberOfMovements = testLength
        simulationTimelimit = movementTime * numberOfMovements + 10
    else:
        simulationTimelimit = testLength

    for GTnumber in range(2, 19):
        env = simpy.Environment()

        inputParams['Locations'] = locations[:GTnumber]

        earth1, graph1, bottleneck1, bottleneck2 = initialize(env,
                                                              'Population Map/gpw_v4_population_count_rev11_2020_15_min.tif',
                                                              'Gateways.csv', 500, inputParams, movementTime, locations)

        progress = env.process(simProgress(simulationTimelimit, env))
        startTime = time.time()
        env.run(simulationTimelimit)
        timeToSim = time.time() - startTime

        if testType == "Rates":

            ratesPath = "./Results/Rates Test/"
            if not os.path.exists(ratesPath):
                # Create a new directory because it does not exist
                os.makedirs(ratesPath)

            data = {"upGSLRates": upGSLRates}
            d = pd.DataFrame(data=data)
            d.to_csv(ratesPath + "UpLinkRates.csv")
            data = {"downGSLRates": downGSLRates}
            d = pd.DataFrame(data=data)
            d.to_csv(ratesPath + "DownLinkRates.csv")
            data = {"interRates": interRates}
            d = pd.DataFrame(data=data)
            d.to_csv(ratesPath + "InterLinkRates.csv")

            plt.clf()
            plt.hist(np.asarray(interRates) / 1e9, cumulative=1, histtype='step', density=True)
            plt.title('CDF - Inter plane ISL data rates')
            plt.ylabel('Empirical CDF')
            plt.xlabel('Data rate [Gbps]')
            plt.savefig(ratesPath + "InterRatesCDF.png")

            plt.clf()
            plt.hist(np.asarray(upGSLRates) / 1e9, cumulative=1, histtype='step', density=True)
            plt.title('CDF - Uplink data rates')
            plt.ylabel('Empirical CDF')
            plt.xlabel('Data rate [Gbps]')
            plt.savefig(ratesPath + "UpLinkRatesCDF.png")

            plt.clf()
            plt.hist(np.asarray(downGSLRates) / 1e9, cumulative=1, histtype='step', density=True)
            plt.title('CDF - Downlink data rates')
            plt.ylabel('Empirical CDF')
            plt.xlabel('Data rate [Gbps]')
            plt.savefig(ratesPath + "DownLinkRatesCDF.png")
            plt.clf()

        else:
            results = getBlockTransmissionStats(timeToSim, inputParams['Locations'], inputParams['Constellation'][0])

            pathBlocks = [[], []]

            first = earth1.gateways[0]
            second = earth1.gateways[1]
            allLatencies = []
            for i, block in enumerate(receivedDataBlocks):
                # get data for all paths in the constellation
                allLatencies.append(block.totLatency)
                if block.source == first and block.destination == second:
                    # get data for the path between the first two gateways
                    pathBlocks[0].append(block.totLatency)
                    pathBlocks[1].append(block)

            xs = [l for l in range(len(allLatencies))]
            plt.figure()
            plt.scatter(xs,allLatencies, c='b')
            plt.ylabel('Latency')
            data = {"latencies": pathBlocks[0]}
            d = pd.DataFrame(data=data)
            d.to_csv(savePath1 + "pathLatencies_{}gateways.csv".format(GTnumber))

            data = {"latencies": allLatencies}
            d = pd.DataFrame(data=data)
            d.to_csv(savePath1 + "allLatencies_{}gateways.csv".format(GTnumber))

            plt.savefig(savePath1 + '{}/{}_gateways.png'.format(testType, GTnumber))
            plt.clf()
            plt.close()

            # add data for percentages plot
            percentages['Queue time'].append(results.meanQueueLatency)  # perQueueLatency)
            percentages['Propagation time'].append(results.meanPropLatency)  # perPropLatency)
            percentages['Transmission time'].append(results.meanTransLatency)  # perTransLatency)
            percentages['GTnumber'].append(GTnumber)

            # get and print information regarding the data block with the highest latency
            longest = 0
            longBlock = None
            for block in receivedDataBlocks:
                longTime = block.getTotalTransmissionTime()
                if longTime > longest:
                    longest = longTime
                    longBlock = block
            print(longBlock.checkPoints)
            print(longBlock.queueLatency)
            print(longest)
            print(longBlock.txLatency)
            print(longBlock.propLatency)
            print(longBlock.totLatency)

            # save received data blocks as "BlocksForPickle" to use in external analysis
            blocks = []
            for block in receivedDataBlocks:
                blocks.append(BlocksForPickle(block))
            np.save("{}blocks_{}".format(blockPath,GTnumber), np.asarray(blocks), allow_pickle=True)

            receivedDataBlocks.clear()
            createdBlocks.clear()
    if testType != "Rates":
        plotLatencies(percentages, pathing, savePath2)


if __name__ == '__main__':
    main()
