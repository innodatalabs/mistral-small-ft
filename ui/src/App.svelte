<script>
  import { filtered, errorsOnly, API_ROOT } from './lib/data.js';
  import { onMount } from 'svelte';

  let showing;
  $: showing = null;

  async function showImage(record, name) {
    showing = name;
  }

</script>
  <div class="header ml-2">
    <b>Count:</b> {$filtered.length}
    <b class="ml-8"><input type="checkbox" bind:checked={$errorsOnly}> Show only errors
  </div>
  <div class="list border border-1 border-gray-700 bg-gray-100">
  {#each $filtered as x }
    <div class="item mb-2 border border-1 border-gray-400 m-1 p-2 rounded bg-white">
      <div class="label text-sm text-gray-600 italic mt-1 mb-0">#{x.id || x.seqno} messages:</div>
      <div class="messages">
        {#each x.messages as m}
        <span class="text-sm text-gray-500 italic">{m.role}:</span>
        {#if Array.isArray(m.content)}
        {#each m.content as c}
          {#if c.type === 'text'}
          {c.text}
          {:else if c.type === 'image'}
          <button class="image" on:click={e => showImage(x, c.image)}>[IMG]&nbsp;</button>
          {:else}
          JSON.stringify(c)
          {/if}
        {/each}
        {:else}
        {m.content}
        {/if}
        {/each}
      </div>
      <div class="label text-sm text-gray-600 italic mt-1 mb-0">expected:</div>
      <div class="expected mb-1">{x.expected}</div>
      {#if x.actual}
      <div class="label text-sm text-gray-600 italic mt-1 mb-0">diff:</div>
      <div class="compare mb-1">
      {#each window.Diff?.diffChars(x.expected, x.actual) as o}<span class={{added: o.added, removed: o.removed}}>{o.value}</span>{/each}
      </div>
      <div class="label text-sm text-gray-600 italic mt-1 mb-0">actual:</div>
      <div class="actual">{x.actual}</div>
      {JSON.stringify(x.actual)}
      {JSON.stringify(x.expected)}
      {/if}
    </div>
  {/each}
  </div>
{#if showing}
<div class="img">
  <div>
    <button aria-label="close" class="float-right mr-4 cursor-pointer" on:click={()=>showing=null}>
      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" class="size-6">
        <path fill-rule="evenodd" d="M5.47 5.47a.75.75 0 0 1 1.06 0L12 10.94l5.47-5.47a.75.75 0 1 1 1.06 1.06L13.06 12l5.47 5.47a.75.75 0 1 1-1.06 1.06L12 13.06l-5.47 5.47a.75.75 0 0 1-1.06-1.06L10.94 12 5.47 6.53a.75.75 0 0 1 0-1.06Z" clip-rule="evenodd" />
      </svg>
    </button>
  </div>
  <img src={`${API_ROOT}/api/image/${showing}`} alt={showing}>
  <!-- <img src="https://fastly.picsum.photos/id/0/5000/3333.jpg?hmac=_j6ghY5fCfSD6tvtcV74zXivkJSPIfR9B8w34XeQmvU"> -->
</div>
{/if}
