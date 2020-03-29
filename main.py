import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from scipy.ndimage.filters import gaussian_filter1d
from matplotlib.ticker import PercentFormatter
from scipy.interpolate import interp1d
import matplotlib.ticker as mticker

plt.rcParams["figure.figsize"] = (9,6)
pal = sns.color_palette("pastel", 10)

def import_data():
    provincial = pd.read_excel(open('data.xlsx', 'rb'), sheet_name='Provincial')
    ontario = pd.read_excel(open('data.xlsx', 'rb'), sheet_name='Ontario')
    tests = pd.read_excel(open('data.xlsx', 'rb'), sheet_name='tests')

    x = mdates.date2num(tests.date)
    f = interp1d(x, tests.tests, 'cubic')
    x = mdates.date2num(provincial.date)
    x = np.linspace(x.min(), x.max(), num = 300)
    provincial['tests'] = f(mdates.date2num(provincial.date))
    provincial['positive_cases'] = provincial.Ontario + provincial.Quebec + provincial.BC + provincial.Manitoba + provincial.Saskatchewan + provincial.Alberta + provincial.Maritimes + provincial.Territories + provincial.Repatriated
    p1 = plt.plot(tests.date, tests.tests, 'o')
    p2 = plt.plot(x, f(x))
    plt.ylabel('Tests Administered', fontsize=10)
    plt.title('COVID-19 Estimated Tests Administered Nationally\nvia Public Health Agency of Canada', fontsize=12)
    plt.legend((p1[0], p2[0]), ('Actual', 'Predicted'), fontsize=8, frameon=False)
    plt.xticks(provincial.date, rotation=45, fontsize=8)
    plt.yticks(fontsize=6)
    plt.tick_params(axis=u'both', which=u'both', length=0)
    # plt.savefig('predicted_tests_canada.png')
    plt.clf()
    plt.cla()

    x = mdates.date2num(ontario.date)
    ontario['pending_ratio'] = ontario.pending / ontario.tested
    f2 = interp1d(x, ontario.pending_ratio, 'cubic')
    x = mdates.date2num(provincial.date)
    x = np.linspace(x.min(), x.max(), num=300)
    provincial['pending_cases'] = f2(mdates.date2num(provincial.date)) * provincial.tests
    provincial['negative_cases'] = provincial.tests - provincial.positive_cases - provincial.pending_cases
    provincial['ratio'] = provincial.positive_cases / (provincial.negative_cases +  provincial.positive_cases)
    p1 = plt.plot(ontario.date, ontario.pending_ratio, 'o')
    p2 = plt.plot(x, f2(x))
    plt.ylabel('Portion of Administered Tests Pending Results', fontsize=10)
    plt.title('COVID-19 Estimated Ratio of Pending Cases\nvia Public Health Agency of Canada & Public Health Ontario', fontsize=12)
    plt.legend((p1[0], p2[0]), ('Actual Ontario', 'Predicted National'), fontsize=8, frameon=False)
    plt.xticks(provincial.date, rotation=45, fontsize=8)
    plt.yticks(fontsize=6)
    plt.tick_params(axis=u'both', which=u'both', length=0)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    # plt.savefig('predicted_ratio_pending.png')
    plt.clf()
    plt.cla()

    return provincial, ontario


def plot_canada(df):
    width = 0.35  # the width of the bars: can also be len(x) sequence

    p1 = plt.bar(df.date, df.positive_cases, width, color=pal[0])
    p2 = plt.bar(df.date, df.pending_cases, width, bottom=df.positive_cases, color=pal[6])
    p3 = plt.bar(df.date, df.negative_cases, width, bottom=df.positive_cases + df.pending_cases, color=pal[2])

    plt.ylabel('Tests Administered', fontsize=10)
    plt.title('COVID-19 Test Results (Canada - Cumulative)\nvia Public Health Agency of Canada', fontsize=12)
    plt.xticks(df.date, rotation=45, fontsize=8)
    plt.yticks(fontsize=8)
    plt.tick_params(axis=u'both', which=u'both', length=0)

    plt2 = plt.twinx()
    ysmoothed = gaussian_filter1d(df.ratio, sigma=2)
    p4 = plt2.plot(df.date, ysmoothed, color=pal[3], marker='', linewidth=2)
    plt2.set_ylabel('Positive Test Rate', fontsize=10)
    plt2.tick_params(axis=u'both', which=u'both', length=0)
    plt2.set_ylim(0, 0.1)
    plt2.set_yticklabels(['{:,.2%}'.format(x) for x in plt2.get_yticks()], fontsize=8)
    plt2.set_xticks(df.date)
    plt.legend((p1[0], p2[0], p3[0], p4[0]), ('Positive (Actual)', 'Pending (Estimated)', 'Negative (Estimated)', 'Positive Test Rate (Includes Probable)'), fontsize=8, frameon=False, loc=2)
    plt2.plot([], [])

    plt.savefig('cumulative_cases_canada.png')
    plt.clf()
    plt.cla()
    return


def plot_canada2(df):
    x = mdates.date2num(df.date)
    z = np.polyfit(x, np.log(df.positive_cases), 2)
    f = np.poly1d(z)
    p1 = plt.plot(df.date, np.exp(f(x)), color=pal[0])
    plt.ylabel('Linear Scale', fontsize=10)
    plt.title('COVID-19 Positive Cases (Canada)\nvia Public Health Ontario', fontsize=12)
    plt.xticks(df.date, rotation=45, fontsize=8)
    plt.yticks(fontsize=8)
    plt.tick_params(axis=u'both', which=u'both', length=0)

    plt2 = plt.twinx()
    # x = mdates.date2num(df.date)
    # z = np.polyfit(x, df.positive, 2)
    # f = np.poly1d(z)
    p2 = plt2.plot(df.date, df.positive_cases, color=pal[6])
    plt2.set_ylabel('Logarithmic Scale', fontsize=10)
    plt2.tick_params(axis=u'both', which=u'both', length=0)
    plt2.set_xticks(df.date)
    plt2.set_yticklabels(df.positive_cases.astype(int), fontsize=8)
    plt2.set_yscale('log')
    plt2.yaxis.set_major_formatter(mticker.ScalarFormatter())
    plt2.yaxis.get_major_formatter().set_scientific(False)
    plt2.yaxis.get_major_formatter().set_useOffset(False)

    plt.legend((p1[0], p2[0]), ('Growth (Linear Scale)', 'Growth (Logarithmic Scale)'), fontsize=8, frameon=False)
    plt2.plot([], [])

    plt.savefig('trends_canada.png')
    plt.clf()
    plt.cla()
    return


def plot_provincial(df):
    width = 0.35  # the width of the bars: can also be len(x) sequence

    p1 = plt.bar(df.date, df.Ontario, width, color=pal[0])
    p2 = plt.bar(df.date, df.Quebec, width, bottom=df.Ontario, color=pal[1])
    p3 = plt.bar(df.date, df.BC, width, bottom=df.Ontario + df.Quebec, color=pal[2])
    p4 = plt.bar(df.date, df.Manitoba, width, bottom=df.Ontario + df.Quebec + df.BC, color=pal[3])
    p5 = plt.bar(df.date, df.Saskatchewan, width, bottom=df.Ontario + df.Quebec + df.BC + df.Manitoba, color=pal[4])
    p6 = plt.bar(df.date, df.Alberta, width, bottom=df.Ontario + df.Quebec + df.BC + df.Manitoba + df.Saskatchewan, color=pal[5])
    p7 = plt.bar(df.date, df.Maritimes, width, bottom=df.Ontario + df.Quebec + df.BC + df.Manitoba + df.Saskatchewan + df.Alberta, color=pal[6])
    p8 = plt.bar(df.date, df.Territories, width, bottom=df.Ontario + df.Quebec + df.BC + df.Manitoba + df.Saskatchewan + df.Alberta + df.Maritimes, color=pal[8])
    p9 = plt.bar(df.date, df.Repatriated, width, bottom=df.Ontario + df.Quebec + df.BC + df.Manitoba + df.Saskatchewan + df.Alberta + df.Maritimes + df.Territories, color=pal[7])

    plt.ylabel('Positive Cases (Includes Probable)', fontsize=10)
    plt.title('COVID-19 Cases by Province (Cumulative)\nvia Public Health Agency of Canada', fontsize=12)
    plt.legend((p1[0], p2[0], p3[0], p4[0], p5[0], p6[0], p7[0], p8[0], p9[0]), ('Ontario', 'Quebec', 'BC', 'Manitoba', 'Saskatchewan', 'Alberta', 'Maritimes', 'Territories', 'Repatriated'), fontsize=6, frameon=False)
    plt.xticks(df.date, rotation=45, fontsize=8)
    plt.yticks(fontsize=8)
    plt.tick_params(axis=u'both', which=u'both', length=0)

    plt2 = plt.twinx()
    x = mdates.date2num(df.date)
    f = interp1d(x, df.Deaths / (df.Ontario + df.Quebec + df.BC + df.Manitoba + df.Saskatchewan + df.Alberta + df.Maritimes + df.Territories + df.Repatriated), 'cubic')
    x = np.linspace(x.min(), x.max(), num=300)
    p10 = plt2.plot(x, f(x), color=pal[3], marker='', linewidth=2)
    plt2.set_ylabel('Death Rate', fontsize=10)
    plt2.tick_params(axis=u'both', which=u'both', length=0)
    plt2.set_ylim(0, 0.1)
    plt2.set_yticklabels(['{:,.2%}'.format(x) for x in plt2.get_yticks()], fontsize=8)
    plt2.set_xticks(df.date)
    plt.legend((p1[0], p2[0], p3[0], p4[0], p5[0], p6[0], p7[0], p8[0], p9[0], p10[0]), ('Ontario', 'Quebec', 'BC', 'Manitoba', 'Saskatchewan', 'Alberta', 'Maritimes', 'Territories', 'Repatriated', 'Death Rate'), loc=2, fontsize=6, frameon=False)
    plt2.plot([], [])

    plt.savefig('cumulative_cases_by_province.png')
    plt.clf()
    plt.cla()
    return


def plot_provincial2(df):
    width = 0.35  # the width of the bars: can also be len(x) sequence

    p1 = plt.bar(df.date, df.Ontario_delta, width, color=pal[0])
    p2 = plt.bar(df.date, df.Quebec_delta, width, bottom=df.Ontario_delta, color=pal[1])
    p3 = plt.bar(df.date, df.BC_delta, width, bottom=df.Ontario_delta + df.Quebec_delta, color=pal[2])
    p4 = plt.bar(df.date, df.Manitoba_delta, width, bottom=df.Ontario_delta + df.Quebec_delta + df.BC_delta, color=pal[3])
    p5 = plt.bar(df.date, df.Saskatchewan_delta, width, bottom=df.Ontario_delta + df.Quebec_delta + df.BC_delta + df.Manitoba_delta, color=pal[4])
    p6 = plt.bar(df.date, df.Alberta_delta, width, bottom=df.Ontario_delta + df.Quebec_delta + df.BC_delta + df.Manitoba_delta + df.Saskatchewan_delta, color=pal[5])
    p7 = plt.bar(df.date, df.Maritimes_delta, width, bottom=df.Ontario_delta + df.Quebec_delta + df.BC_delta + df.Manitoba_delta + df.Saskatchewan_delta + df.Alberta_delta, color=pal[6])
    p8 = plt.bar(df.date, df.Territories_delta, width, bottom=df.Ontario_delta + df.Quebec_delta + df.BC_delta + df.Manitoba_delta + df.Saskatchewan_delta + df.Alberta_delta + df.Maritimes_delta, color=pal[8])
    p9 = plt.bar(df.date, df.Repatriated_delta, width, bottom=df.Ontario_delta + df.Quebec_delta + df.BC_delta + df.Manitoba_delta + df.Saskatchewan_delta + df.Alberta_delta + df.Maritimes_delta + df.Territories_delta, color=pal[7])

    plt.ylabel('Positive Cases (Includes Probable)', fontsize=10)
    plt.title('COVID-19 Cases by Province (Daily)\nvia Public Health Agency of Canada', fontsize=12)
    plt.legend((p1[0], p2[0], p3[0], p4[0], p5[0], p6[0], p7[0], p8[0], p9[0]), ('Ontario', 'Quebec', 'BC', 'Manitoba', 'Saskatchewan', 'Alberta', 'Maritimes', 'Territories', 'Repatriated'), fontsize=6, frameon=False)
    plt.xticks(df.date, rotation=45, fontsize=8)
    plt.yticks(fontsize=8)
    plt.tick_params(axis=u'both', which=u'both', length=0)

    plt2 = plt.twinx()
    df2 = df.drop(df.index[0])
    x = mdates.date2num(df2.date)
    f = interp1d(x, df2.Deaths_Delta / (df2.Ontario_delta + df2.Quebec_delta + df2.BC_delta + df2.Manitoba_delta + df2.Saskatchewan_delta + df2.Alberta_delta + df2.Maritimes_delta + df2.Territories_delta), 'cubic')
    x = np.linspace(x.min(), x.max(), num=300)
    p10 = plt2.plot(x, f(x), color=pal[3], marker='', linewidth=2)
    plt2.set_ylabel('Death Rate', fontsize=10)
    plt2.tick_params(axis=u'both', which=u'both', length=0)
    plt2.set_ylim(0, 0.1)
    plt2.set_yticklabels(['{:,.2%}'.format(x) for x in plt2.get_yticks()], fontsize=8)
    plt2.set_xticks(df.date)
    plt.legend((p1[0], p2[0], p3[0], p4[0], p5[0], p6[0], p7[0], p8[0], p9[0], p10[0]), ('Ontario', 'Quebec', 'BC', 'Manitoba', 'Saskatchewan', 'Alberta', 'Maritimes', 'Territories', 'Repatriated', 'Death Rate'), loc=2, fontsize=6, frameon=False)
    plt2.plot([], [])
    # plt.show()
    plt.savefig('daily_cases_by_province.png')
    plt.clf()
    plt.cla()
    return


def plot_ontario(df):
    width = 0.35  # the width of the bars: can also be len(x) sequence

    p1 = plt.bar(df.date, df.positive, width, color=pal[0])
    p2 = plt.bar(df.date, df.pending, width, bottom=df.positive, color=pal[6])
    p3 = plt.bar(df.date, df.resolved, width, bottom=df.positive + df.pending, color=pal[1])
    p4 = plt.bar(df.date, df.negative, width, bottom=df.positive + df.pending + df.resolved, color=pal[2])
    p5 = plt.bar(df.date, df.deceased, width, bottom=df.positive + df.pending + df.resolved + df.negative, color=pal[7])

    plt.ylabel('Tests Administered', fontsize=10)
    plt.title('COVID-19 Test Results (Ontario - Cumulative)\nvia Public Health Ontario', fontsize=12)
    plt.xticks(df.date, rotation=45, fontsize=8)
    plt.yticks(fontsize=6)
    plt.tick_params(axis=u'both', which=u'both',length=0)

    plt2 = plt.twinx()
    ysmoothed = gaussian_filter1d(df.ratio, sigma=2)
    p6 = plt2.plot(df.date, ysmoothed, color=pal[3], marker='', linewidth=2)
    plt2.set_ylabel('Positive Test Rate', fontsize=10)
    plt2.tick_params(axis=u'both', which=u'both', length=0)
    plt2.set_ylim(0,0.1)
    plt2.set_yticklabels(['{:,.2%}'.format(x) for x in plt2.get_yticks()], fontsize=8)
    plt2.set_xticks(df.date)
    plt.legend((p1[0], p2[0], p3[0], p4[0], p5[0], p6[0]), ('Positive', 'Pending', 'Resolved', 'Negative', 'Deceased', 'Positive Test Rate (Excluding Pending Results)'), fontsize=8, frameon=False, loc=2)
    plt2.plot([], [])

    plt.savefig('cumulative_cases_ontario.png')
    plt.clf()
    plt.cla()
    return


def plot_ontario3(df):
    width = 0.35  # the width of the bars: can also be len(x) sequence

    p1 = plt.bar(df.date, df.delta_positive, width, color=pal[0])
    p3 = plt.bar(df.date, df.delta_resolved, width, bottom=df.delta_positive, color=pal[1])
    p4 = plt.bar(df.date, df.delta_negative, width, bottom=df.delta_positive + df.delta_resolved, color=pal[2])
    p5 = plt.bar(df.date, df.delta_deceased, width, bottom=df.delta_positive + df.delta_resolved + df.delta_negative, color=pal[7])

    plt.ylabel('Test Results Confirmed', fontsize=10)
    plt.title('COVID-19 Test Results (Ontario - Daily)\nvia Public Health Ontario', fontsize=12)
    plt.xticks(df.date, rotation=45, fontsize=8)
    plt.yticks(fontsize=6)
    plt.tick_params(axis=u'both', which=u'both',length=0)

    plt2 = plt.twinx()
    df2 = df.drop(df.index[0])
    # x = mdates.date2num(df2.date)
    # f = interp1d(x, df2.delta_ratio, 'cubic')
    # x = np.linspace(x.min(), x.max(), num=300)
    ysmoothed = gaussian_filter1d(df2.delta_ratio, sigma=2)
    p6 = plt2.plot(df2.date, ysmoothed, color=pal[3], marker='', linewidth=2)
    plt2.set_ylabel('% of Test Results Returning Positive', fontsize=10)
    plt2.tick_params(axis=u'both', which=u'both', length=0)
    plt2.set_ylim(0,0.1)
    plt2.set_yticklabels(['{:,.2%}'.format(x) for x in plt2.get_yticks()], fontsize=8)
    plt2.set_xticks(df.date)
    plt.legend((p1[0], p3[0], p4[0], p5[0], p6[0]), ('New Positive Cases', 'New Resolved Cases', 'New Negative Cases', 'New Deceased Cases', '% of New Cases Returning Positive'), fontsize=8, frameon=False, loc=2)
    plt2.plot([], [])

    # plt.show()
    plt.savefig('daily_cases_ontario.png')
    plt.clf()
    plt.cla()
    return


def plot_ontario2(df):

    x = mdates.date2num(df.date)
    z = np.polyfit(x, np.log(df.positive), 2)
    f = np.poly1d(z)
    p1 = plt.plot(df.date, np.exp(f(x)), color=pal[0])
    plt.ylabel('Linear Scale', fontsize=10)
    plt.title('COVID-19 Positive Cases (Ontario)\nvia Public Health Ontario', fontsize=12)
    plt.xticks(df.date, rotation=45, fontsize=8)
    plt.yticks(fontsize=8)
    plt.tick_params(axis=u'both', which=u'both',length=0)

    plt2 = plt.twinx()
    # x = mdates.date2num(df.date)
    # z = np.polyfit(x, df.positive, 2)
    # f = np.poly1d(z)
    p2 = plt2.plot(df.date, df.positive, color=pal[6])
    plt2.set_ylabel('Logarithmic Scale', fontsize=10)
    plt2.tick_params(axis=u'both', which=u'both', length=0)
    plt2.set_xticks(df.date)
    plt2.set_yticklabels(df.positive.astype(int), fontsize=8)
    plt2.set_yscale('log')
    plt2.yaxis.set_major_formatter(mticker.ScalarFormatter())
    plt2.yaxis.get_major_formatter().set_scientific(False)
    plt2.yaxis.get_major_formatter().set_useOffset(False)

    plt.legend((p1[0], p2[0]), ('Growth (Linear Scale)', 'Growth (Logarithmic Scale)'), fontsize=8, frameon=False)
    plt2.plot([], [])

    plt.savefig('trends_ontario.png')
    plt.clf()
    plt.cla()
    return


if __name__ == '__main__':
    dfs = import_data()
    # plot_canada(dfs[0])
    # plot_provincial(dfs[0])
    # plot_provincial2(dfs[0])
    # plot_ontario(dfs[1])
    # plot_ontario3(dfs[1])
    plot_ontario2(dfs[1])
    plot_canada2(dfs[0])

