# 克隆自聚宽文章：https://www.joinquant.com/post/10272
# 标题：【量化课堂】RSRS(阻力支撑相对强度)择时策略（下）
# 作者：JoinQuant量化课堂

# 导入函数库
import statsmodels.api as sm
#from pandas.stats.api import ols

# 初始化函数，设定基准等等
def initialize(context):
    # 设定上证指数作为基准
    set_benchmark('000300.XSHG')
    # 开启动态复权模式(真实价格)
    set_option('use_real_price', True)
    # 输出内容到日志 log.info()
    log.info('初始函数开始运行且全局只运行一次')
    # 过滤掉order系列API产生的比error级别低的log
    # log.set_level('order', 'error')

    ### 股票相关设定 ###
    # 股票类每笔交易时的手续费是：买入时佣金万分之三，卖出时佣金万分之三加千分之一印花税, 每笔交易佣金最低扣5块钱
    set_order_cost(OrderCost(close_tax=0.001, open_commission=0.0003, close_commission=0.0003, min_commission=5), type='stock')

    ## 运行函数（reference_security为运行时间的参考标的；传入的标的只做种类区分，因此传入'000300.XSHG'或'510300.XSHG'是一样的）
      # 开盘前运行
    run_daily(before_market_open, time='before_open', reference_security='000300.XSHG')
      # 开盘时运行
    run_daily(market_open, time='open', reference_security='000300.XSHG')
      # 收盘后运行
    run_daily(after_market_close, time='after_close', reference_security='000300.XSHG')

    # 设置RSRS指标中N, M的值
    g.N = 18
    g.M = 1100
    g.init = True

    # 要操作的股票：平安银行（g.为全局变量）
    g.security = '000300.XSHG'

    # 买入阈值
    g.buy = 0.7
    g.sell = -0.7
    g.ans = []
    g.ans_rightdev= []

    # 计算2005年1月5日至回测开始日期的RSRS斜率指标
    prices = get_price(g.security, '2005-01-05', context.previous_date, '1d', ['high', 'low']).dropna()
    highs = prices.high
    lows = prices.low
    g.ans = []
    for i in range(len(highs))[g.N:]:
        data_high = highs.iloc[i-g.N+1:i+1]
        data_low = lows.iloc[i-g.N+1:i+1]
        X = sm.add_constant(data_low)
        model = sm.OLS(data_high,X)
        results = model.fit()
        g.ans.append(results.params.low)
        #计算r2
        g.ans_rightdev.append(results.rsquared)


## 开盘前运行函数
def before_market_open(context):
    # 输出运行时间
    log.info('函数运行时间(before_market_open)：'+str(context.current_dt.time()))

    # 给微信发送消息（添加模拟交易，并绑定微信生效）
    send_message('美好的一天~')


## 开盘时运行函数
def market_open(context):
    log.info('函数运行时间(market_open):'+str(context.current_dt.time()))
    security = g.security
    # 取得当前的现金
    cash = context.portfolio.available_cash

    # 填入各个日期的RSRS斜率值

    security = g.security
    beta=0
    r2=0

    if g.init:
        g.init = False
    else:
        # RSRS斜率指标定义
        prices = attribute_history(security, g.N, '1d', ['high', 'low'],fq=None) #指数无复权,个股应该使用前复权
        highs = prices.high
        lows = prices.low
        X = sm.add_constant(lows)
        model = sm.OLS(highs, X)
        beta = model.fit().params.low
        g.ans.append(beta)
        #计算r2
        r2=model.fit().rsquared
        g.ans_rightdev.append(r2)

    # 计算标准化的RSRS指标
    # 计算均值序列
    section = g.ans[-g.M:]
    # 计算均值序列
    mu = np.mean(section)
    # 计算标准化RSRS指标序列
    sigma = np.std(section)
    zscore = (section[-1]-mu)/sigma
    #计算右偏RSRS标准分
    zscore_rightdev= zscore*beta*r2


    # 如果上一时间点的RSRS斜率大于买入阈值, 则全仓买入
    if zscore_rightdev > g.buy:
        # 记录这次买入
        log.info("标准化RSRS斜率大于买入阈值, 买入 %s" % (security))
        # 用所有 cash 买入股票
        order_value(security, cash)
    # 如果上一时间点的RSRS斜率小于卖出阈值, 则空仓卖出
    elif zscore_rightdev < g.sell and context.portfolio.positions[security].closeable_amount > 0:
        # 记录这次卖出
        log.info("标准化RSRS斜率小于卖出阈值, 卖出 %s" % (security))
        # 卖出所有股票,使这只股票的最终持有量为0
        order_target(security, 0)

## 收盘后运行函数
def after_market_close(context):
    log.info(str('函数运行时间(after_market_close):'+str(context.current_dt.time())))
    #得到当天所有成交记录
    trades = get_trades()
    for _trade in trades.values():
        log.info('成交记录：'+str(_trade))
    log.info('一天结束')
    log.info('##############################################################')
