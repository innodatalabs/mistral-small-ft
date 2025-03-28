import { writable, derived } from 'svelte/store';

async function fetchJSONL(url) {
    const reply = await fetch(url);
    if (reply.status !== 200) {
        throw new Error(`fetch failed for ${url}`);
    }
    const text = await reply.text();
    const records = text.split('\n').filter(x => x !== '').map(x => JSON.parse(x));

    let seqno = 0;
    for (const r of records) {
        seqno += 1;
        r.seqno = seqno;
    }
    return records;
}

export const errorsOnly = writable(false);

export const API_ROOT = ''; // 'http://localhost:8000';

export const data = (() => {
    const { subscribe, set } = writable([]);

    fetchJSONL(API_ROOT + '/api/data').then(dataset => set(dataset));

    return { subscribe }
})();


export const filtered = derived([errorsOnly, data], ([errorsOnly, data]) => {
    return data.filter(x => !errorsOnly || (x.actual && x.expected !== x.actual));
}, []);
