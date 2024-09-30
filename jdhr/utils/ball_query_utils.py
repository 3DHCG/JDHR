import jittor as jt

def ball_query(unknown, known, k, radius):
    ''' find k neighbors for unknown array from known array

    Args:
        
        unknown (var): shape [b, n, c]
        known (var): shape [b, m, c]
        k (int)

    '''
    b, n, c = unknown.shape
    _, m, _ = known.shape
    dists2 = jt.empty((b, n, k), dtype="float")
    idx = jt.empty((b, n, k), dtype="int")
    src = '''
__inline_static__
@python.jittor.auto_parallel(2, block_num=256)
void ball_query_kernel(int b, int batch_index, int n, int index, int m,
                        const float *__restrict__ unknown,
                        const float *__restrict__ known,
                        float *__restrict__ dist2,
                        int *__restrict__ idx) {

#define K %s
#define radius %s
    unknown += batch_index * n * 3;
    known += batch_index * m * 3;
    dist2 += batch_index * n * K;
    idx += batch_index * n * K;
    int j = index;
    {
        float ux = unknown[j * 3 + 0];
        float uy = unknown[j * 3 + 1];
        float uz = unknown[j * 3 + 2];

        float tmp_dist[K];
        int tmp_idx[K];
        #pragma unroll
        for (int i=0; i<K; i++) tmp_dist[i] = 0.0;
        for (int i=0; i<K; i++) tmp_idx[i] = -1;
        for (int k = 0; k < m; ++k) {
            float x = known[k * 3 + 0];
            float y = known[k * 3 + 1];
            float z = known[k * 3 + 2];
            float d = (ux - x) * (ux - x) + (uy - y) * (uy - y) + (uz - z) * (uz - z);
            
            if(d < radius){
                int first = -1;
                #pragma unroll
                for (int i=0; i<K; i++)
                    if (tmp_idx[i]==-1)
                        first = i;
                tmp_dist[first] = d;
                tmp_idx[first] = k;
            }
        }
        #pragma unroll
        for (int i=0; i<K; i++) {
            dist2[j * K + i] = tmp_dist[K-1-i];
            idx[j * K + i] = tmp_idx[K-1-i];
        }
    }
}
    ball_query_kernel(in0->shape[0], 0, in0->shape[1], 0, in1->shape[1], in0_p, in1_p, out0_p, out1_p);
    ''' % (k,radius)
    return jt.code([unknown, known], [dists2, idx],
    cpu_src=src,
    cuda_src=src)