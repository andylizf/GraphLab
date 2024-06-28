// Modified from https://www.acwing.com/blog/content/30378/

#include <bits/stdc++.h>

using namespace std;

const int N = 110, M = 2410, inf = 0x3f3f3f3f;

int n, m, S, T;
int h[N], e[M], ne[M], idx;
int q[N], dep[N], cur[N], dg[N];
double c[M];
bool st[N];

struct Node{
    int a, b;
}edge[M];

void add(int u, int v, double w1, double w2)
{
    e[idx] = v, c[idx] = w1, ne[idx] = h[u], h[u] = idx ++ ;
    e[idx] = u, c[idx] = w2, ne[idx] = h[v], h[v] = idx ++ ;
}

bool bfs()
{
    memset(dep, -1, sizeof dep);
    int hh = 0, tt = 0;
    q[0] = S, dep[S] = 0;
    cur[S] = h[S];
    while (hh <= tt)
    {
        int t = q[hh ++ ];
        for (int i = h[t]; ~i; i = ne[i])
        {
            int j = e[i];
            if (dep[j] == -1 && c[i] > 0)
            {
                dep[j] = dep[t] + 1;
                cur[j] = h[j];
                if (j == T)return true;
                q[++ tt] = j;
            }
        }
    }
    return false;
}

double find(int u, double limit)
{
    if (u == T)return limit;
    double flow = 0;
    for (int i = cur[u]; ~i && flow < limit; i = ne[i])
    {
        cur[u] = i;
        int j = e[i];
        if (dep[j] == dep[u] + 1 && c[i] > 0)
        {
            double t = find(j, min(c[i], limit - flow));
            if (!t)dep[j] = -1;
            c[i] -= t, c[i ^ 1] += t, flow += t;
        }
    }
    return flow;
}

void build(double g)
{
    idx = 0;
    memset(h, -1, sizeof h);
    for (int i = 0; i < m; i ++ )add(edge[i].a, edge[i].b, 1, 1);
    for (int i = 1; i <= n; i ++ )add(S, i, m, 0), add(i, T, m + 2 * g - dg[i], 0);
}

double dinic(double g)
{
    build(g);
    double res = 0, flow;
    while (bfs())while (flow = find(S, inf))res += flow;
    return res;
}

void dfs(int u)
{
    st[u] = true;
    for (int i = h[u]; ~i; i = ne[i])
    {
        int j = e[i];
        if (!st[j] && c[i] > 0)
            dfs(j);
    }
}

int main()
{
    scanf("%d%d", &n, &m);
    S = 0, T = n + 1;

    for (int i = 0; i < m; i ++ )
    {
        scanf("%d%d", &edge[i].a, &edge[i].b);
        dg[edge[i].a] ++ , dg[edge[i].b] ++ ;
    }

    double l = 0, r = m;
    while (r - l > 1e-8)
    {
        double mid = (l + r) / 2;
        double t = dinic(mid);
        if (n * m - t > 0)l = mid;
        else r = mid;
    }

    dinic(l);
    dfs(S);

    for (int i = 1; i <= n; i ++ )
        if (st[i])printf("%d\n", i);

    return 0;
}